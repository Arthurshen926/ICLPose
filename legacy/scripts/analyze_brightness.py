"""Analyze brightness and color variation between best/worst views."""
import torchvision.io as tio
import os

source = "dataset/OldHospital"
worst = ["seq8/frame00013.png", "seq4/frame00054.png", "seq8/frame00010.png", 
         "seq4/frame00053.png", "seq4/frame00012.png", "seq8/frame00016.png"]
best = ["seq8/frame00094.png", "seq8/frame00093.png", "seq4/frame00002.png",
        "seq4/frame00001.png", "seq8/frame00092.png", "seq8/frame00102.png"]

print("=== WORST VIEWS (brightness) ===")
for name in worst:
    img = tio.read_image(os.path.join(source, name)).float() / 255.0
    print(f"  {name}: mean={img.mean():.4f} std={img.std():.4f}")

print("\n=== BEST VIEWS (brightness) ===")
for name in best:
    img = tio.read_image(os.path.join(source, name)).float() / 255.0
    print(f"  {name}: mean={img.mean():.4f} std={img.std():.4f}")

print("\n=== Per-channel R/G/B means ===")
for label, views in [("WORST", worst[:3]), ("BEST", best[:3])]:
    print(f"  [{label}]")
    for name in views:
        img = tio.read_image(os.path.join(source, name)).float() / 255.0
        print(f"    {name}: R={img[0].mean():.4f} G={img[1].mean():.4f} B={img[2].mean():.4f}")
