import * as vscode from "vscode";
import { ChildProcess, spawn } from "child_process";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface JsonRpcRequest {
  jsonrpc: "2.0";
  id: number;
  method: string;
  params: Record<string, unknown>;
}

interface JsonRpcResponse {
  jsonrpc: "2.0";
  id: number;
  result?: unknown;
  error?: { code: number; message: string; data?: unknown };
}

type ConnectionState = "disconnected" | "connecting" | "connected" | "error";

// ---------------------------------------------------------------------------
// Extension state
// ---------------------------------------------------------------------------

let serverProcess: ChildProcess | null = null;
let outputChannel: vscode.OutputChannel;
let statusBarItem: vscode.StatusBarItem;
let requestId = 0;
let connectionState: ConnectionState = "disconnected";
let responseBuffer = "";

const pendingRequests = new Map<
  number,
  { resolve: (value: JsonRpcResponse) => void; reject: (reason: Error) => void }
>();

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

export function activate(context: vscode.ExtensionContext): void {
  outputChannel = vscode.window.createOutputChannel("QUAD Agent");
  context.subscriptions.push(outputChannel);

  statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 100);
  statusBarItem.command = "quad.detectHardware";
  context.subscriptions.push(statusBarItem);
  updateStatusBar("disconnected");
  statusBarItem.show();

  startServer(context);

  context.subscriptions.push(
    vscode.commands.registerCommand("quad.detectHardware", cmdDetectHardware),
    vscode.commands.registerCommand("quad.convertModel", cmdConvertModel),
    vscode.commands.registerCommand("quad.profileWorkload", cmdProfileWorkload),
    vscode.commands.registerCommand("quad.orchestrateWorkload", cmdOrchestrateWorkload),
    vscode.commands.registerCommand("quad.generateCode", cmdGenerateCode)
  );
}

export function deactivate(): void {
  killServer();
}

// ---------------------------------------------------------------------------
// Server management
// ---------------------------------------------------------------------------

function getConfig(): vscode.WorkspaceConfiguration {
  return vscode.workspace.getConfiguration("quad");
}

function startServer(context: vscode.ExtensionContext): void {
  const config = getConfig();
  const command = config.get<string>("serverCommand", "quad-server");
  const adapterMode = config.get<string>("adapterMode", "mock");

  updateStatusBar("connecting");
  outputChannel.appendLine(`[QUAD] Starting server: ${command} --adapter-mode ${adapterMode}`);

  try {
    serverProcess = spawn(command, ["--adapter-mode", adapterMode], {
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env },
    });
  } catch (err: unknown) {
    handleServerNotFound(command);
    return;
  }

  if (!serverProcess || !serverProcess.stdout || !serverProcess.stdin) {
    handleServerNotFound(command);
    return;
  }

  serverProcess.stdout.on("data", (data: Buffer) => {
    responseBuffer += data.toString();
    processResponseBuffer();
  });

  serverProcess.stderr?.on("data", (data: Buffer) => {
    outputChannel.appendLine(`[QUAD server stderr] ${data.toString().trim()}`);
  });

  serverProcess.on("error", (err: Error) => {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") {
      handleServerNotFound(command);
    } else {
      handleServerCrash(err.message, context);
    }
  });

  serverProcess.on("exit", (code: number | null, signal: string | null) => {
    const reason = signal ? `signal ${signal}` : `code ${code}`;
    outputChannel.appendLine(`[QUAD] Server exited (${reason})`);
    if (connectionState === "connected" || connectionState === "connecting") {
      handleServerCrash(`Server exited unexpectedly (${reason})`, context);
    }
  });

  // Send initialize to confirm connectivity
  updateStatusBar("connected");
  outputChannel.appendLine("[QUAD] Server started successfully");
}

function killServer(): void {
  if (serverProcess) {
    serverProcess.kill("SIGTERM");
    serverProcess = null;
  }
  rejectAllPending("Server shut down");
  updateStatusBar("disconnected");
}

function handleServerNotFound(command: string): void {
  updateStatusBar("error");
  connectionState = "error";
  const msg = `QUAD server "${command}" not found. Install with: pip install quad-server`;
  outputChannel.appendLine(`[QUAD] ERROR: ${msg}`);
  vscode.window
    .showErrorMessage(msg, "Open Install Docs")
    .then((choice) => {
      if (choice === "Open Install Docs") {
        vscode.env.openExternal(
          vscode.Uri.parse("https://github.com/qualcomm/quad#installation")
        );
      }
    });
}

function handleServerCrash(reason: string, context: vscode.ExtensionContext): void {
  updateStatusBar("error");
  connectionState = "error";
  rejectAllPending(reason);
  outputChannel.appendLine(`[QUAD] ERROR: Server crashed — ${reason}`);
  vscode.window
    .showErrorMessage(`QUAD server crashed: ${reason}`, "Restart Server")
    .then((choice) => {
      if (choice === "Restart Server") {
        killServer();
        startServer(context);
      }
    });
}

function rejectAllPending(reason: string): void {
  for (const [id, handler] of pendingRequests) {
    handler.reject(new Error(reason));
  }
  pendingRequests.clear();
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------

function updateStatusBar(state: ConnectionState): void {
  connectionState = state;
  switch (state) {
    case "disconnected":
      statusBarItem.text = "$(circle-outline) QUAD";
      statusBarItem.tooltip = "QUAD: Disconnected";
      break;
    case "connecting":
      statusBarItem.text = "$(sync~spin) QUAD";
      statusBarItem.tooltip = "QUAD: Connecting...";
      break;
    case "connected":
      statusBarItem.text = "$(check) QUAD";
      statusBarItem.tooltip = "QUAD: Connected";
      break;
    case "error":
      statusBarItem.text = "$(error) QUAD";
      statusBarItem.tooltip = "QUAD: Error — click to retry";
      break;
  }
}

// ---------------------------------------------------------------------------
// JSON-RPC transport
// ---------------------------------------------------------------------------

function sendRequest(toolName: string, args: Record<string, unknown>): Promise<JsonRpcResponse> {
  return new Promise((resolve, reject) => {
    if (!serverProcess || !serverProcess.stdin) {
      reject(new Error("Server not connected"));
      return;
    }

    const id = ++requestId;
    const request: JsonRpcRequest = {
      jsonrpc: "2.0",
      id,
      method: "tools/call",
      params: { name: toolName, arguments: args },
    };

    const payload = JSON.stringify(request) + "\n";
    pendingRequests.set(id, { resolve, reject });

    outputChannel.appendLine(`[QUAD] --> ${JSON.stringify(request, null, 2)}`);

    serverProcess.stdin.write(payload, (err) => {
      if (err) {
        pendingRequests.delete(id);
        reject(new Error(`Failed to write to server stdin: ${err.message}`));
      }
    });

    // Timeout after 60 seconds
    setTimeout(() => {
      if (pendingRequests.has(id)) {
        pendingRequests.delete(id);
        reject(new Error("Request timed out (60s)"));
      }
    }, 60_000);
  });
}

function processResponseBuffer(): void {
  // The server sends newline-delimited JSON
  const lines = responseBuffer.split("\n");
  // Keep the last (possibly incomplete) chunk
  responseBuffer = lines.pop() ?? "";

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    let response: JsonRpcResponse;
    try {
      response = JSON.parse(trimmed) as JsonRpcResponse;
    } catch (err: unknown) {
      outputChannel.appendLine(`[QUAD] JSON parse error: ${(err as Error).message}`);
      outputChannel.appendLine(`[QUAD] Raw line: ${trimmed}`);
      continue;
    }

    outputChannel.appendLine(`[QUAD] <-- ${JSON.stringify(response, null, 2)}`);

    if (response.id !== undefined && pendingRequests.has(response.id)) {
      const handler = pendingRequests.get(response.id)!;
      pendingRequests.delete(response.id);
      handler.resolve(response);
    }
  }
}

// ---------------------------------------------------------------------------
// Helper to display results
// ---------------------------------------------------------------------------

function displayResult(title: string, response: JsonRpcResponse): void {
  outputChannel.show(true);
  outputChannel.appendLine("");
  outputChannel.appendLine(`━━━ ${title} ━━━`);

  if (response.error) {
    outputChannel.appendLine(`ERROR [${response.error.code}]: ${response.error.message}`);
    if (response.error.data) {
      outputChannel.appendLine(JSON.stringify(response.error.data, null, 2));
    }
    vscode.window.showErrorMessage(`QUAD: ${response.error.message}`);
  } else {
    outputChannel.appendLine(JSON.stringify(response.result, null, 2));
    vscode.window.showInformationMessage(`QUAD: ${title} completed`);
  }

  outputChannel.appendLine("");
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

async function cmdDetectHardware(): Promise<void> {
  const config = getConfig();
  const adapterMode = config.get<string>("adapterMode", "mock");

  try {
    const response = await sendRequest("detect_hardware", {
      adapter_mode: adapterMode,
    });
    displayResult("Detect Hardware", response);
  } catch (err: unknown) {
    vscode.window.showErrorMessage(`QUAD: ${(err as Error).message}`);
  }
}

async function cmdConvertModel(): Promise<void> {
  const modelPath = await vscode.window.showInputBox({
    title: "QUAD: Convert Model",
    prompt: "Path to input model (e.g., model.onnx, model.pt)",
    placeHolder: "/path/to/model.onnx",
    ignoreFocusOut: true,
  });
  if (!modelPath) {
    return;
  }

  const targetFormat = await vscode.window.showQuickPick(
    ["onnx", "tflite", "qnn", "dlc"],
    {
      title: "QUAD: Target Format",
      placeHolder: "Select target model format",
    }
  );
  if (!targetFormat) {
    return;
  }

  const targetPlatform = await vscode.window.showQuickPick(
    ["hexagon-npu", "adreno-gpu", "kryo-cpu", "cloud-cpu"],
    {
      title: "QUAD: Target Platform",
      placeHolder: "Select target hardware platform",
    }
  );
  if (!targetPlatform) {
    return;
  }

  const quantization = await vscode.window.showQuickPick(
    ["none", "int8", "int4", "fp16", "mixed"],
    {
      title: "QUAD: Quantization",
      placeHolder: "Select quantization scheme (optional)",
    }
  );

  try {
    const response = await sendRequest("convert_model", {
      model_path: modelPath,
      target_format: targetFormat,
      target_platform: targetPlatform,
      quantization: quantization ?? "none",
    });
    displayResult("Convert Model", response);
  } catch (err: unknown) {
    vscode.window.showErrorMessage(`QUAD: ${(err as Error).message}`);
  }
}

async function cmdProfileWorkload(): Promise<void> {
  const modelPath = await vscode.window.showInputBox({
    title: "QUAD: Profile Workload",
    prompt: "Path to compiled model or workload directory",
    placeHolder: "/path/to/compiled_model",
    ignoreFocusOut: true,
  });
  if (!modelPath) {
    return;
  }

  const targetPlatform = await vscode.window.showQuickPick(
    ["hexagon-npu", "adreno-gpu", "kryo-cpu"],
    {
      title: "QUAD: Profile Target",
      placeHolder: "Select hardware target for profiling",
    }
  );
  if (!targetPlatform) {
    return;
  }

  const iterations = await vscode.window.showInputBox({
    title: "QUAD: Profiling Iterations",
    prompt: "Number of inference iterations for profiling",
    value: "100",
    validateInput: (val) => {
      const n = parseInt(val, 10);
      return isNaN(n) || n <= 0 ? "Must be a positive integer" : null;
    },
  });
  if (!iterations) {
    return;
  }

  try {
    const response = await sendRequest("profile_workload", {
      model_path: modelPath,
      target_platform: targetPlatform,
      iterations: parseInt(iterations, 10),
    });
    displayResult("Profile Workload", response);
  } catch (err: unknown) {
    vscode.window.showErrorMessage(`QUAD: ${(err as Error).message}`);
  }
}

async function cmdOrchestrateWorkload(): Promise<void> {
  const workloadPath = await vscode.window.showInputBox({
    title: "QUAD: Orchestrate Workload",
    prompt: "Path to workload configuration (YAML/TOML)",
    placeHolder: "/path/to/workload.yaml",
    ignoreFocusOut: true,
  });
  if (!workloadPath) {
    return;
  }

  const strategy = await vscode.window.showQuickPick(
    ["latency-optimal", "throughput-optimal", "power-efficient", "balanced"],
    {
      title: "QUAD: Orchestration Strategy",
      placeHolder: "Select scheduling strategy",
    }
  );
  if (!strategy) {
    return;
  }

  const targetPlatforms = await vscode.window.showQuickPick(
    ["hexagon-npu", "adreno-gpu", "kryo-cpu"],
    {
      title: "QUAD: Target Platforms",
      placeHolder: "Select available hardware (multi-select)",
      canPickMany: true,
    }
  );
  if (!targetPlatforms || targetPlatforms.length === 0) {
    return;
  }

  try {
    const response = await sendRequest("orchestrate_workload", {
      workload_path: workloadPath,
      strategy,
      target_platforms: targetPlatforms,
    });
    displayResult("Orchestrate Workload", response);
  } catch (err: unknown) {
    vscode.window.showErrorMessage(`QUAD: ${(err as Error).message}`);
  }
}

async function cmdGenerateCode(): Promise<void> {
  const description = await vscode.window.showInputBox({
    title: "QUAD: Generate Code",
    prompt: "Describe the inference code to generate",
    placeHolder: "e.g., Run MobileNetV2 on Hexagon NPU with camera input",
    ignoreFocusOut: true,
  });
  if (!description) {
    return;
  }

  const language = await vscode.window.showQuickPick(
    ["python", "cpp", "java", "kotlin"],
    {
      title: "QUAD: Language",
      placeHolder: "Select output language",
    }
  );
  if (!language) {
    return;
  }

  const targetPlatform = await vscode.window.showQuickPick(
    ["hexagon-npu", "adreno-gpu", "kryo-cpu", "cloud-cpu"],
    {
      title: "QUAD: Target Platform",
      placeHolder: "Select target hardware platform",
    }
  );
  if (!targetPlatform) {
    return;
  }

  try {
    const response = await sendRequest("generate_code", {
      description,
      language,
      target_platform: targetPlatform,
    });

    displayResult("Generate Code", response);

    // If successful, offer to open the generated code in a new editor
    if (response.result && typeof response.result === "object") {
      const result = response.result as Record<string, unknown>;
      const code = result.code as string | undefined;
      if (code) {
        const action = await vscode.window.showInformationMessage(
          "QUAD: Code generated. Open in editor?",
          "Open",
          "Dismiss"
        );
        if (action === "Open") {
          const langMap: Record<string, string> = {
            python: "python",
            cpp: "cpp",
            java: "java",
            kotlin: "kotlin",
          };
          const doc = await vscode.workspace.openTextDocument({
            content: code,
            language: langMap[language] ?? "plaintext",
          });
          await vscode.window.showTextDocument(doc);
        }
      }
    }
  } catch (err: unknown) {
    vscode.window.showErrorMessage(`QUAD: ${(err as Error).message}`);
  }
}
