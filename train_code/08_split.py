img_ls = '''![baemin (1)_4](https://user-images.githubusercontent.com/56625356/79941301-f64a8700-849e-11ea-9b60-3d95c44d546f.jpg)
![baemin (1)_0](https://user-images.githubusercontent.com/56625356/79941302-f77bb400-849e-11ea-8e9a-50178a3b2b02.jpg)
![baemin (1)_1](https://user-images.githubusercontent.com/56625356/79941303-f77bb400-849e-11ea-8b71-767a2c6e06dd.jpg)
![baemin (1)_2](https://user-images.githubusercontent.com/56625356/79941304-f8144a80-849e-11ea-9a3e-7cc6722ef1bf.jpg)
![baemin (1)_3](https://user-images.githubusercontent.com/56625356/79941307-f8ace100-849e-11ea-9b85-6f2597ab5c0d.jpg)'''.split("(")

urls = []
for i in range(0,len(img_ls),2):
    print(img_ls[i].split("\n")[0].replace(")","").replace("![baemin",""))