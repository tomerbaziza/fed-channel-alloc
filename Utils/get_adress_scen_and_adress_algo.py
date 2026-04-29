import os 

def get_adress_scen_and_adress_algo(script_path):
    parent_path = os.path.dirname(script_path)
    grandparent = os.path.dirname(parent_path)
    gradgradparent = os.path.dirname(grandparent)
    address_scen = gradgradparent + '\Simulation\SimAladdin\Source\Preferences\MBN'
    address_algo = script_path 
    
    return address_scen, address_algo