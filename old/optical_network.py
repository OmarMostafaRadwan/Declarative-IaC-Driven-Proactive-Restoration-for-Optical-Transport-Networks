from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.link import TCLink
import threading
import time
import json
import os

def create_nsfnet():
    net = Mininet(controller=RemoteController, switch=OVSKernelSwitch, link=TCLink)
    
    print("--- Building NSFNET Topology (14 Nodes, 21 Links) ---")
    c0 = net.addController('c0', controller=RemoteController, ip='127.0.0.1', port=6633)

    # 1. Create 14 Nodes (backbone routers)
    # Naming convention: s1 to s14
    switches = {}
    for i in range(1, 15):
        switches[f's{i}'] = net.addSwitch(f's{i}')
    
    # 2. Add Hosts (One at West Coast, One at East Coast)
    h1 = net.addHost('h1', ip='10.0.0.1') # Connected to CA (s1)
    h2 = net.addHost('h2', ip='10.0.0.2') # Connected to NY (s14)
    
    net.addLink(h1, switches['s1'])
    net.addLink(h2, switches['s14'])

    # 3. Create NSFNET Backbone Links (The "Continental" connections)
    # Tuples of (NodeA, NodeB, Delay) representing fiber distance
    links = [
        ('s1', 's2', '10ms'), ('s1', 's3', '10ms'), ('s1', 's8', '10ms'),
        ('s2', 's3', '10ms'), ('s2', 's4', '10ms'),
        ('s3', 's6', '10ms'),
        ('s4', 's5', '10ms'), ('s4', 's9', '10ms'),
        ('s5', 's6', '10ms'), ('s5', 's7', '10ms'),
        ('s6', 's10', '10ms'), ('s6', 's13', '10ms'),
        ('s7', 's8', '10ms'), ('s8', 's9', '10ms'),
        ('s9', 's10', '10ms'), ('s9', 's12', '10ms'), ('s9', 's14', '10ms'),
        ('s10', 's11', '10ms'), ('s11', 's12', '10ms'), ('s11', 's14', '10ms'),
        ('s12', 's13', '10ms'), ('s13', 's14', '10ms')
    ]

    for (u, v, d) in links:
        net.addLink(switches[u], switches[v], delay=d, bw=1000)

    print("--- Starting Network ---")
    net.start()
    
    # 4. Define Paths (K-Shortest Paths)
    # Path A: The "Primary" (Northern Route)
    # s1 -> s2 -> s4 -> s5 -> s6 -> s13 -> s14
    path_primary = ['s1', 's2', 's4', 's5', 's6', 's13', 's14']
    
    # Path B: The "Backup" (Southern Route)
    # s1 -> s8 -> s9 -> s14
    path_backup = ['s1', 's8', 's9', 's14']

    # Helper to enforce a path
    def set_path(path_nodes):
        # Reset all links to down first (simplified simulation)
        # In reality, we just change flows, but for simulation visual we toggle links
        print(f"Configuring Path: {path_nodes}")
        # (This logic would be more complex in real SDN, simplified for visual)

    # 5. DevOps Monitor Thread
    def monitor_trigger():
        print("--- DevOps Monitor Active: Watching Terraform State... ---")
        last_route = "primary"
        while True:
            if os.path.exists("network_state.json"):
                try:
                    with open("network_state.json", "r") as f:
                        data = json.load(f)
                    current_route = data.get("route")
                    
                    if current_route != last_route:
                        print(f"\n!!! NSFNET RE-ROUTING TO: {current_route.upper()} !!!")
                        if current_route == "backup":
                            print(f"Activating Southern Route: {path_backup}")
                            # Simulate switchover
                        else:
                            print(f"Activating Northern Route: {path_primary}")
                        last_route = current_route
                except:
                    pass
            time.sleep(1)

    t = threading.Thread(target=monitor_trigger)
    t.daemon = True
    t.start()

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    create_nsfnet()