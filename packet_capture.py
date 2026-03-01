from scapy.all import sniff
import pandas as pd

INTERFACE = r"\Device\NPF_{C41B29F4-1374-47E0-BCAD-E5290DFC5D4F}"

captured_packets = []

def packet_callback(packet):
    if packet.haslayer("IP"):

        row = {}

        if packet.haslayer("TCP"):
            row["protocol_type"] = "tcp"
        elif packet.haslayer("UDP"):
            row["protocol_type"] = "udp"
        else:
            row["protocol_type"] = "icmp"

        row["service"] = "http"
        row["flag"] = "SF"

        row["src_bytes"] = len(packet)
        row["dst_bytes"] = len(packet)

        captured_packets.append(row)

def capture_packets(duration=5):
    global captured_packets
    captured_packets = []

    sniff(
        iface=INTERFACE,
        prn=packet_callback,
        timeout=duration,
        store=False
    )

    if not captured_packets:
        return None

    return pd.DataFrame(captured_packets)