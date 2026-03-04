# Dynamic 3D Object Removal

動的物体を除去する 3D 点群デモ。

## 触ってみる（3Dデモ）

### GitHub Pages

- https://rsasaki0109.github.io/dynamic-3d-object-removal/
- https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d_standalone.html
- https://rsasaki0109.github.io/dynamic-3d-object-removal/demo/index_3d.html

### ローカルで試す

```bash
git clone git@github.com:rsasaki0109/dynamic-3d-object-removal.git
cd dynamic-3d-object-removal/demo
python3 -m pip install -r requirements.txt
python3 run_demo.py
python3 -m http.server 4173
```

ブラウザで `http://127.0.0.1:4173/index_3d.html` を開く。
