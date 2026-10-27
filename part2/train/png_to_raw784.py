from PIL import Image
import sys

in_path  = sys.argv[1]
out_path = sys.argv[2]

img = Image.open(in_path).convert("L").resize((28,28))
# اگر لازم شد این خط رو فعال کن تا سفید/مشکی درست شه:
# img = Image.eval(img, lambda p: 255 - p)

open(out_path, "wb").write(img.tobytes())
print("ok")
