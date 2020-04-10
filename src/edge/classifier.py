from PIL import Image

import edge_tpu
import tflite_runtime.interpreter as tflite
import platform

TOP_K       = 1
THRESHOLD   = 0.0
REPEAT      = 5

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def load_labels(path, encoding='utf-8'):
  """Loads labels from file (with or without index numbers).

  Args:
    path: path to label file.
    encoding: label file encoding.
  Returns:
    Dictionary mapping indices to labels.
  """
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
    if not lines:
      return {}

    if lines[0].split(' ', maxsplit=1)[0].isdigit():
      pairs = [line.split(' ', maxsplit=1) for line in lines]
      return {int(index): label.strip() for index, label in pairs}
    else:
      return {index: line.strip() for index, line in enumerate(lines)}

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

class Classifier():
    def __init__(self, model_path, label_path):
        self.model_path = model_path
        self.label_path = label_path
    
    def initialize(self):
        self.labels = load_labels(PATH_LABELS) if PATH_LABELS else {}
        self.interpreter = make_interpreter(PATH_MODEL)
        self.interpreter.allocate_tensors()

        self.input_size = classify.input_size(interpreter)
    
    def predict(self, image):
        w, h = img.shape[:2]
        if(w != input_size[0] or h != input_size[1]) :
            image = cv2.resize(img, (w, h))
        
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        classify.set_input(interpreter, image)
        interpreter.invoke()

        classes = classify.get_output(interpreter, TOP_K, THRESHOLD)
        print('%.1fms' % (inference_time * 1000))
        for klass in classes:
            print('%d %s: %.5f' % (klass.id, labels[klass.id], klass.score))

if __name__ == '__main__':
  main()
