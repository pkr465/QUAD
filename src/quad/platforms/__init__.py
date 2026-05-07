"""QUAD Platform Layer — abstracts device communication per platform."""

from quad.platforms.base import Platform, DeviceInfo
from quad.platforms.windows import WindowsPlatform
from quad.platforms.linux import LinuxPlatform
from quad.platforms.android import AndroidPlatform


def get_platform(platform_type: str) -> Platform:
    """Get a platform instance by name ('windows', 'linux', 'android')."""
    platforms = {
        "windows": WindowsPlatform,
        "linux": LinuxPlatform,
        "android": AndroidPlatform,
    }
    cls = platforms.get(platform_type)
    if cls is None:
        raise ValueError(
            f"Unknown platform: '{platform_type}'. Available: {list(platforms.keys())}"
        )
    return cls()


__all__ = ["AndroidPlatform", "DeviceInfo", "LinuxPlatform", "Platform", "WindowsPlatform", "get_platform"]
