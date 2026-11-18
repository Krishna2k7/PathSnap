import json

with open("reports/trip_00000007_svg_data.json") as f:
    d = json.load(f)

gt_path = "M " + " L ".join(f"{x},{y}" for x, y in d["gt"])
pred_path = "M " + " L ".join(f"{x},{y}" for x, y in d["pred"])
gps_circles = "".join(f'<circle cx="{x}" cy="{y}" r="2" fill="#3b82f6" opacity="0.6"/>' for x, y in d["gps"])

with open("reports/svg_elements.json", "w") as out:
    json.dump({"gt_path": gt_path, "pred_path": pred_path, "gps_circles": gps_circles}, out)

print("SVG elements ready.")
