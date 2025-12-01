# encoding=gbk
import tensorrt as trt
import numpy as np
import os
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from imutils import paths
from tqdm import tqdm


class TrtModel:
    def __init__(self, engine_path, max_batch_size=1, dtype=None):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.context = self.engine.create_execution_context()
        # 从引擎获取输入数据类型
        if dtype is None:
            for i in range(self.engine.num_bindings):
                if self.engine.binding_is_input(i):
                    self.dtype = trt.nptype(self.engine.get_binding_dtype(i))
                    break
        else:
            self.dtype = dtype
        # 重新组织显存分配逻辑
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        return trt_runtime.deserialize_cuda_engine(engine_data)

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            binding = i  # 使用索引访问
            dims = self.engine.get_binding_shape(binding)

            # --- 修复 1: 智能处理动态 Batch ---
            # 如果维度中有 -1，替换为 max_batch_size 来分配最大可能的显存
            if dims[0] == -1:
                dims[0] = self.max_batch_size

            size = trt.volume(dims)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # 分配 Host 和 Device 内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            # 记录绑定信息，保存 shape 以便后续 reshape
            binding_info = {
                'host': host_mem,
                'device': device_mem,
                'shape': dims,  # 保存原始计算出的维度 (MaxBatch, 300, 4)
                'index': i
            }

            if self.engine.binding_is_input(binding):
                inputs.append(binding_info)
            else:
                outputs.append(binding_info)

        return inputs, outputs, bindings, stream

    def __call__(self, x: np.ndarray):
        # x shape: (Batch, 3, 640, 640)
        batch_size = x.shape[0]

        # 1. 准备输入数据
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0]['host'][:x.size], x.ravel().astype(self.inputs[0]['host'].dtype))

        # 2. 拷贝 Host -> Device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # 3. 设置动态 Input Shape (关键步骤)
        # 必须在 execute_async_v2 之前设置
        self.context.set_binding_shape(self.inputs[0]['index'], x.shape)

        # 4. 执行推理 (修复 2: 使用 V2 接口)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 5. 拷贝 Device -> Host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()

        # 6. 解析输出 (修复 3: 自动 Reshape，不再硬编码 150)
        results = []
        for out in self.outputs:
            # 获取该输出绑定的最大维度 (例如 [1, 300, 4])
            shape = list(out['shape'])
            shape[0] = batch_size  # 修正为当前实际 Batch

            # 截取有效数据并 Reshape
            valid_size = trt.volume(shape)
            data = out['host'][:valid_size].reshape(shape)
            results.append(data)

        return results


if __name__ == "__main__":
    w, h = 640, 640
    path = r'./images/baofeng'
    trt_engine_path = r'./onnx/best-trt.engine'

    if not os.path.exists('./results_trt'):
        os.makedirs('./results_trt')

    palette = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}
    label_map = {0: 'TuAn', 1: 'LV'}

    # 初始化模型
    print("Loading Engine...")
    model = TrtModel(trt_engine_path, max_batch_size=1)

    pic_paths = list(paths.list_images(path))

    for pic_path in tqdm(pic_paths):
        basename = os.path.basename(pic_path)
        img_raw = cv2.imread(pic_path)
        if img_raw is None: continue

        H, W = img_raw.shape[:2]
        img_bak = img_raw.copy()

        # --- 预处理 (保持与 4_onnxrun 一致) ---
        img = cv2.resize(img_raw, (w, h))
        # 注意：4_onnxrun 中没有做 /255.0，这里也保持不做
        # 如果 ONNX 实际上是 implicitly normalized，这里 TRT 可能需要手动 /255.0
        # 如果跑出来框很多但位置不对，请尝试解开下面这行的注释：
        # img = img.astype(np.float32) / 255.0

        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # (1, 3, 640, 640)

        # --- 推理 ---
        out = model(img)

        # --- 解析输出 ---
        # 这里的 out[0] 和 out[1] 取决于 Engine 导出时的顺序
        # 如果结果很怪，尝试交换 out[0] 和 out[1]
        dets = out[0][0]  # 期望: [300, 4]
        scores_probs = out[1][0]  # 期望: [300, num_classes]

        # 如果发现 dets 维度是 [300, num_classes] 而 scores 是 [300, 4]，则交换它们
        if dets.shape[-1] != 4:
            dets, scores_probs = scores_probs, dets

        detections = []
        for i in range(len(dets)):
            bbox = dets[i]
            probs = scores_probs[i]

            class_id = np.argmax(probs)
            confidence = probs[class_id]

            if confidence > 0.35:
                cx, cy, w_obj, h_obj = bbox

                # 反归一化
                cx = cx * W
                cy = cy * H
                w_obj = w_obj * W
                h_obj = h_obj * H

                x1 = int(cx - w_obj / 2)
                y1 = int(cy - h_obj / 2)
                x2 = int(cx + w_obj / 2)
                y2 = int(cy + h_obj / 2)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                detections.append({'bbox': [x1, y1, x2, y2], 'class_id': class_id, 'conf': confidence})

        # --- 可视化 ---
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            conf = det['conf']

            color = palette.get(class_id, (0, 255, 0))
            cv2.rectangle(img_bak, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label_map.get(class_id, str(class_id))}: {conf:.2f}"
            cv2.putText(img_bak, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        output_path = os.path.join('./results_trt', basename)
        cv2.imwrite(output_path, img_bak)