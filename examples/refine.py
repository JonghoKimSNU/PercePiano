import json

a = "examples/model_prediction_comparison.json"
with open(a) as f:
    data = json.load(f)
print(data.keys())

for key in data:
    if key == "better_labels":
        continue
    for i in range(len(data[key])):
        data[key][i] = round(data[key][i], 4)

# round the values to 4 decimal places
with open(a.replace(".json", "_1.json"), "w") as f:
    json.dump(data, f, indent=4)