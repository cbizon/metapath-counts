# A quick script to look into input KG to see if we really have a particular set
# of nodes or edges

import json
#edgefile="/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/84e6183aaeef2a8c/edges.jsonl"
#nodefile="/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/84e6183aaeef2a8c/nodes.jsonl"
edgefile="../translator_kg/Jan_20_filtered_nonredundant/edges.jsonl"
nodefile="../translator_kg/Jan_20_filtered_nonredundant/nodes.jsonl"
phens = set()
chems = set()
dises = set()
with open(nodefile,"r") as inf:
    for line in inf:
        if "Disease" in line or "Phenotypic" in line:
            l = json.loads(line)
            if "biolink:Disease" in l['category']:
                dises.add(l['id'])
            if "biolink:PhenotypicFeature" in l['category']:
                phens.add(l['id'])
print(f"{len(phens)} Phens found")
print(f"{len(dises)} Diseases found")


count = 0
with open(edgefile,"r") as inf, open("fuckups.txt", "w") as outf:
    for line in inf:
        if "treats" in line:
#        if True:
            l = json.loads(line)
            if l['subject'] in phens and l['object'] in dises:
                outf.write(line)
                print(line)
                count+=1
                print(count)
            
