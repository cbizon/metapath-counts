# A quick script to look into input KG to see if we really have a particular set
# of nodes or edges

import json
#edgefile="/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/84e6183aaeef2a8c/edges.jsonl"
#nodefile="/projects/stars/Data_services/biolink3/graphs/Baseline_Nonredundant/84e6183aaeef2a8c/nodes.jsonl"
edgefile="../../../translator_kg/Feb_13_filtered_nonredundant/edges.jsonl"
nodefile="../../../translator_kg/Feb_13_filtered_nonredundant/nodes.jsonl"
t1 = set()
t2 = set()
n1 = "Drug"
n2 = "Agent"
with open(nodefile,"r") as inf:
    for line in inf:
        if n1 in line or n2 in line:
            l = json.loads(line)
            if f"biolink:{n1}" in l['category']:
                t1.add(l['id'])
            if f"biolink:{n2}" in l['category']:
                t2.add(l['id'])
print(f"{len(t1)} {n1} found")
print(f"{len(t2)} {n2} found")


p="member_of"
count = 0
with open(edgefile,"r") as inf, open("fuckups.txt", "w") as outf:
    for line in inf:
        #if p in line:
        if True:
            l = json.loads(line)
            if l['subject'] in t1 and l['object'] in t2:
                outf.write(line)
                print(line)
                count+=1
                print("F",count)
            if l['subject'] in t2 and l['object'] in t1:
                outf.write("reverse")
                outf.write(line)
                print(line)
                count_r += 1
                print("R",count_r)
            
