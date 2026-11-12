"""Pure server resource normalization."""
def get_server_current_host(server):
    if isinstance(server, dict):
        return (
            server.get("OS-EXT-SRV-ATTR:host") or
            server.get("compute_host") or
            server.get("OS-EXT-SRV-ATTR:hypervisor_hostname") or
            server.get("hypervisor_hostname") or
            server.get("host")
        )

    return (
        getattr(server, "OS-EXT-SRV-ATTR:host", None) or
        getattr(server, "compute_host", None) or
        getattr(server, "OS-EXT-SRV-ATTR:hypervisor_hostname", None) or
        getattr(server, "hypervisor_hostname", None) or
        getattr(server, "host", None)
    )


def get_server_status(server):
    if isinstance(server, dict):
        return str(server.get("status") or server.get("Status") or "UNKNOWN").upper()

    return str(getattr(server, "status", None) or "UNKNOWN").upper()


