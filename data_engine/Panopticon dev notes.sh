C:\Users\tempo\Desktop\Python> .\Scripts\pyinstaller.exe --onefile ..\Panopticon\main.py
pyinstaller --onefile main.py --name demo
C:\Users\tempo\Desktop\Python> .\python.exe ..\Panopticon\main.py
C:\Users\tempo\Desktop\Python> ./python -m pip install opencv-python transformers[vision] torch[cuda] pyside6 scipy pyinstaller timm "optimum[openvino]" openvino ultralytics pycocotools platformdirs supervision datumaro py-cpuinfo clip onnxslim onnxruntime-gpu

# filter dataset
czkawka_cli similar-images --directories /path/to/your/images --image-hash-type DoubleGradient --image-resize-filter Lanczos3 --image-hash-size 16 --delete-similar-except-biggest

# compile
source /home/master-andreas/panopticon/test_env/bin/activate && rm -fR demo_dir demo.spec && pyinstaller --onedir panopticon/main.py --name pan_opticon --distpath demo_dir && ./demo_dir/pan_opticon/pan_opticon
source /home/master-andreas/panopticon/test_env/bin/activate && rm -fR demo_dir demo.spec && pyinstaller --onefile panopticon/main.py --name pan_opticon --distpath demo_dir && ./demo_dir/pan_opticon
source /home/master-andreas/panopticon/test_env/bin/activate && rm -fR demo_dir demo.spec && pyinstaller --onedir panopticon/pan_annotator.py --name pan_annotator --distpath demo_dir && ./demo_dir/pan_annotator/pan_annotator



