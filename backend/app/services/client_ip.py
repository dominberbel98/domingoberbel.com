"""The visitor's IP address, as seen from behind the ingress.

`request.client.host` is the peer of the TCP connection. In Azure Container
Apps that peer is the environment's ingress proxy, so every visitor arrived as
the same 100.100.x.x address. Two things depended on it and were quietly wrong:

* the per-IP rate limit was one bucket shared by the whole internet, so a
  single visitor hammering the chat locked everyone else out;
* the IP stored with each logged question, and the one sent to Turnstile,
  identified nobody.

The proxy reports the real peer by appending it to `X-Forwarded-For`. The header
is a list and the *client* controls everything already in it when the request
arrives, so the leftmost entry — the one most code reads — can be forged to get
a fresh rate-limit bucket per request. The entries that can be trusted are the
ones our own proxies appended, which are the rightmost ones. With one proxy in
front (the Container Apps ingress), that is the last entry.
"""

from __future__ import annotations

from ipaddress import ip_address

from fastapi import Request

from app.config import settings


def _clean(value: str) -> str | None:
    """A bare IP from one X-Forwarded-For entry, or None if it is not one."""
    value = value.strip().strip('"')
    if value.startswith("["):                      # [2001:db8::1]:443
        value = value[1:].split("]", 1)[0]
    elif value.count(":") == 1:                    # 203.0.113.7:51234
        value = value.split(":", 1)[0]
    try:
        return str(ip_address(value))
    except ValueError:
        return None


def client_ip(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"
    hops = settings.trusted_proxy_hops
    if hops <= 0:
        return peer
    entries = [e for e in request.headers.get("x-forwarded-for", "").split(",") if e.strip()]
    if len(entries) < hops:
        # Fewer entries than proxies we expect: the request did not come
        # through the ingress (local run, health probe). Trust the socket.
        return peer
    return _clean(entries[-hops]) or peer
