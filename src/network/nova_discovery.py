"""
Lumos Nervous System - mDNS Discovery Service

Handles Zeroconf (mDNS) service discovery for the Lumos Pi 4 server.
Allows Flutter clients to discover the device on the local network without hardcoded IPs.
"""

import asyncio
import logging
import socket
import subprocess
from typing import Optional
from zeroconf import ServiceInfo, Zeroconf

logger = logging.getLogger(__name__)


class LumosDiscovery:
    """
    mDNS service responder for Lumos network discovery.

    Registers the Lumos service on the local network using Zeroconf,
    enabling automatic discovery by mobile clients on the Pi's hotspot.
    """

    SERVICE_TYPE = "_lumos._tcp.local."
    SERVICE_NAME = "Lumos Server"
    HOSTNAME     = "pi-four.local"   # FIXED: was "lumos.local"
    PORT         = 8000              # FIXED: was 5000

    def __init__(self):
        self.zeroconf: Optional[Zeroconf] = None
        self.service_info: Optional[ServiceInfo] = None
        self._running = False

    async def start_discovery(self) -> None:
        """
        Start the mDNS service registration asynchronously.

        Registers the Lumos service on the local network, making it discoverable
        by clients searching for _lumos._tcp.local. services.
        """
        if self._running:
            logger.warning("Discovery service is already running")
            return

        try:
            local_ip = self._get_local_ip()
            if not local_ip:
                raise RuntimeError("Could not determine local IP address")

            self.zeroconf = Zeroconf()

            self.service_info = ServiceInfo(
                type_=self.SERVICE_TYPE,
                name=f"{self.SERVICE_NAME}.{self.SERVICE_TYPE}",
                addresses=[socket.inet_aton(local_ip)],
                port=self.PORT,
                properties={
                    "version": "3.1",
                    "hostname": self.HOSTNAME,
                },
            )

            await asyncio.get_event_loop().run_in_executor(
                None, self.zeroconf.register_service, self.service_info
            )

            self._running = True
            logger.info(f"Lumos mDNS service registered at {local_ip}:{self.PORT}")

        except Exception as e:
            logger.error(f"Failed to start mDNS discovery: {e}")
            await self.stop_discovery()
            raise

    async def stop_discovery(self) -> None:
        """Stop the mDNS service registration and clean up resources."""
        if not self._running:
            return

        try:
            if self.zeroconf and self.service_info:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.zeroconf.unregister_service, self.service_info
                )
            if self.zeroconf:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.zeroconf.close
                )
        except Exception as e:
            logger.error(f"Error stopping mDNS discovery: {e}")
        finally:
            self.zeroconf = None
            self.service_info = None
            self._running = False
            logger.info("Lumos mDNS service unregistered")

    def _get_local_ip(self) -> Optional[str]:
        """
        Get the local IP address of wlan0 (phone hotspot interface).
        Falls back to hostname resolution if wlan0 is not up yet.
        """
        # FIXED: don't connect to 8.8.8.8 — no internet on phone hotspot.
        # Instead read wlan0 IP directly from the system.
        try:
            result = subprocess.run(
                ["ip", "-4", "addr", "show", "wlan0"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("inet "):
                    ip = line.split()[1].split("/")[0]
                    if not ip.startswith("127."):
                        return ip
        except Exception:
            pass

        # Fallback: hostname resolution
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return None
