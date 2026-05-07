"""QUAD UDO — User-Defined Operations for custom neural-network layers.

Provides the tools needed to define, package, compile, quantize, and deploy
custom operators that run on CPU, GPU, DSP, and HTP via Qualcomm SNPE/QAIRT SDK.

Supports BOTH config formats:
  - JSON UDO Config (legacy): {"UdoPackage_0": {"Operators": [...]}}
  - XML OpDef Config (newer): <OpDefCollection> with SupplementalOpDefList

Typical workflow::

    from quad.udo import UDOManager, UDOPackage, UDORuntime, UDOConfig

    mgr = UDOManager()                          # mock mode (no SDK needed)

    pkg = mgr.generate_package(
        config_json="Softmax_Htp.json",
        output_dir="/tmp/udo_pkgs",
    )

    dlc = mgr.convert_model(
        model_path="model.onnx",
        output_dlc="/tmp/model.dlc",
        udo_config="Softmax_Htp.json",
    )

    libs = mgr.compile_package(pkg.package_dir, runtime="cpu_android")

    q_dlc = mgr.quantize_with_udo(
        input_dlc=dlc,
        output_dlc="/tmp/model_q.dlc",
        input_list="inputs.txt",
        reg_lib_path=pkg.get_reg_lib("cpu"),
    )
"""

from quad.udo.manager import UDOManager
from quad.udo.package import UDOConfig, UDOPackage, UDORuntime

__all__ = [
    "UDOConfig",
    "UDOManager",
    "UDOPackage",
    "UDORuntime",
]
