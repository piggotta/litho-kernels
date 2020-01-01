import json

if __name__ == '__main__':

    source = [{
        'type': 'circ',
        'sigma': 0.8,
        'ctr': [0, 0]
         }]

    lithosys = {
        'source': source,
        'NA': 0.8,
        'n': 1,
        'delta_z': 1,
        'fmax': 2,
        'Lxy': 8,
        'Nxy': 256,
        'num': 5
        }

    with open('example_sys.json', 'w') as f:
        f.write(json.dumps(lithosys, indent=4))
