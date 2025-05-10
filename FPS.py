import warnings

warnings.filterwarnings('ignore')
import argparse
import os
import time
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
import subprocess
import re
import platform
import psutil
from threading import Thread
from queue import Queue
import logging
from pathlib import Path
import cv2
from ultralytics.yolo.utils.torch_utils import select_device
from ultralytics.nn.tasks import attempt_load_weights

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class PowerMonitor:
    """Monitor system power consumption"""

    def __init__(self, gpu_id=0):
        self.os_type = platform.system()
        self.gpu_id = gpu_id
        self.power_data = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.data_queue = Queue()

    def start_monitoring(self):
        """Start power monitoring"""
        self.is_monitoring = True
        self.power_data = []
        self.monitor_thread = Thread(target=self._monitor_power)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logging.info("Power monitoring started")

    def stop_monitoring(self):
        """Stop power monitoring"""
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2)
            logging.info("Power monitoring stopped")
        return self.get_power_stats()

    def get_power_stats(self):
        """Get power consumption statistics"""
        if not self.power_data:
            return {"avg": 0, "max": 0, "min": 0, "std": 0}
        power_array = np.array(self.power_data)
        return {
            "avg": np.mean(power_array),
            "max": np.max(power_array),
            "min": np.min(power_array),
            "std": np.std(power_array)
        }

    def _monitor_power(self):
        """Main monitoring thread function"""
        while self.is_monitoring:
            try:
                if self.os_type == "Windows":
                    self._monitor_windows_power()
                elif self.os_type == "Linux":
                    self._monitor_linux_power()
                elif self.os_type == "Darwin":
                    self._monitor_macos_power()
                else:
                    logging.warning(f"Unsupported OS: {self.os_type}")
                    break
                time.sleep(1)
            except Exception as e:
                logging.error(f"Power monitoring error: {e}")
                time.sleep(5)

    def _monitor_windows_power(self):
        try:
            if torch.cuda.is_available():
                result = subprocess.check_output(
                    ["nvidia-smi", f"--id={self.gpu_id}", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    universal_newlines=True
                )
                gpu_power = float(result.strip())
                self.power_data.append(gpu_power)
            else:
                cpu_percent = psutil.cpu_percent(interval=None)
                estimated_power = cpu_percent * 0.05
                self.power_data.append(estimated_power)
        except Exception as e:
            logging.warning(f"Windows power monitoring error: {e}")

    def _monitor_linux_power(self):
        try:
            if torch.cuda.is_available():
                result = subprocess.check_output(
                    ["nvidia-smi", f"--id={self.gpu_id}", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    universal_newlines=True
                )
                gpu_power = float(result.strip())
                self.power_data.append(gpu_power)
            else:
                try:
                    subprocess.check_output(["powertop", "--csv=/tmp/powertop_temp.csv", "--time=1"],
                                            stderr=subprocess.DEVNULL)
                    with open("/tmp/powertop_temp.csv", "r") as f:
                        for line in f:
                            if "Total" in line and "Power" in line:
                                match = re.search(r"(\d+\.\d+)\s*W", line)
                                if match:
                                    total_power = float(match.group(1))
                                    self.power_data.append(total_power)
                                    break
                except Exception:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    estimated_power = cpu_percent * 0.05
                    self.power_data.append(estimated_power)
        except Exception as e:
            logging.warning(f"Linux power monitoring error: {e}")


class RealImageDataset(torch.utils.data.Dataset):
    """Dataset for loading real images"""

    def __init__(self, img_dir, img_size=(640, 640)):
        self.img_files = [str(p) for p in Path(img_dir).rglob('*.*')
                          if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        if not self.img_files:
            raise ValueError(f"No images found in {img_dir}")
        self.img_size = img_size
        logging.info(f"Loaded {len(self.img_files)} images from {img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        try:
            img = cv2.imread(self.img_files[index])
            if img is None:
                raise ValueError(f"Failed to read {self.img_files[index]}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
            img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
            return img, torch.zeros(1, 5)  # Return dummy labels
        except Exception as e:
            logging.warning(f"Error loading {self.img_files[index]}: {e}")
            return torch.zeros((3, *self.img_size)), torch.zeros(1, 5)


def get_weight_size(path):
    try:
        stats = os.stat(path)
        return f'{stats.st_size / (1024  **  2):.1f}'
    except OSError as e:
        logging.error(f"Error getting weight size: {e}")
        return "N/A"


def warmup_model(model, device, dataloader, iterations=200):
    logging.info("Starting warmup...")
    iter_loader = iter(dataloader)
    for _ in tqdm(range(iterations), desc='Warmup'):
        try:
            inputs, _ = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            inputs, _ = next(iter_loader)
        inputs = inputs.to(device)
        if inputs.dtype != torch.float32:
            inputs = inputs.half()
        model(inputs)


def model_latency_test(model, device, dataloader, iterations=1000):
    logging.info("Testing latency with real images...")
    time_arr = []
    iter_loader = iter(dataloader)

    for _ in tqdm(range(iterations), desc='Latency Test'):
        try:
            inputs, _ = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            inputs, _ = next(iter_loader)

        inputs = inputs.to(device)
        if inputs.dtype != torch.float32:
            inputs = inputs.half()

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start_time = time.time()
        model(inputs)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        end_time = time.time()
        time_arr.append(end_time - start_time)
    return np.mean(time_arr), np.std(time_arr)


def measure_baseline_power(power_monitor, duration=30):
    logging.info(f"Starting baseline power measurement ({duration} seconds)...")
    power_monitor.start_monitoring()
    time.sleep(duration)
    stats = power_monitor.stop_monitoring()
    logging.info(f"Baseline power: {stats['avg']:.2f}W (min: {stats['min']:.2f}W, max: {stats['max']:.2f}W)")
    return stats


def measure_inference_power(model, device, dataloader, power_monitor, baseline_power, iterations=100):
    logging.info(f"Starting inference power measurement ({iterations} iterations)...")
    power_monitor.start_monitoring()
    iter_loader = iter(dataloader)

    for _ in tqdm(range(iterations), desc='Inference Power Test'):
        try:
            inputs, _ = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            inputs, _ = next(iter_loader)

        inputs = inputs.to(device)
        if inputs.dtype != torch.float32:
            inputs = inputs.half()

        model(inputs)
        time.sleep(0.01)

    stats = power_monitor.stop_monitoring()
    delta_power = stats['avg'] - baseline_power

    logging.info(f"Inference power: {stats['avg']:.2f}W (Δ {delta_power:.2f}W)")
    return stats


def calculate_efficiency(fps, total_power, baseline_power):
    """计算能效比（新增基线参数）"""
    delta_power = total_power - baseline_power
    if delta_power <= 0:
        logging.warning(f"Invalid power delta: {delta_power:.2f}W (total: {total_power:.2f}W, baseline: {baseline_power:.2f}W)")
        return 0.0
    return fps / delta_power


def different_batch_sizes_test(model, device, dataloader, power_monitor, baseline_power, batch_sizes):
    logging.info("\n" + "=" * 50)
    logging.info("Testing different batch sizes with baseline deduction")
    results = []

    for batch_size in batch_sizes:
        logging.info(f"\nTesting batch size: {batch_size}")
        current_dataloader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers,
            pin_memory=True
        )


        warmup_model(model, device, current_dataloader, 100)


        mean_latency, std_latency = model_latency_test(model, device, current_dataloader, 500)
        fps = batch_size / mean_latency


        power_stats = measure_inference_power(model, device, current_dataloader, power_monitor, baseline_power, 50)


        efficiency = calculate_efficiency(fps, power_stats['avg'], baseline_power)

        results.append({
            'batch_size': batch_size,
            'latency': mean_latency,
            'latency_std': std_latency,
            'fps': fps,
            'total_power': power_stats['avg'],
            'delta_power': power_stats['avg'] - baseline_power,
            'efficiency': efficiency
        })

        logging.info(f"Batch {batch_size}: "
                     f"Latency: {mean_latency:.5f}s ± {std_latency:.5f}s | "
                     f"FPS: {fps:.1f} | "
                     f"Δ Power: {results[-1]['delta_power']:.2f}W | "
                     f"Efficiency: {efficiency:.2f} FPS/W")

    return results


def main(opt):

    device = select_device(opt.device)
    gpu_id = 0 if opt.device.isdigit() and int(opt.device) >= 0 else None


    power_monitor = PowerMonitor(gpu_id=gpu_id)


    model = attempt_load_weights(opt.weights, device=device, fuse=True)
    model = model.to(device).eval()
    if opt.half and device.type != 'cpu':
        model = model.half()


    dataset = RealImageDataset(opt.data_dir, tuple(opt.imgs))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch,
        num_workers=opt.workers,
        pin_memory=True,
        shuffle=False
    )


    warmup_model(model, device, dataloader, opt.warmup)


    baseline_stats = measure_baseline_power(power_monitor, opt.baseline_duration)
    baseline_power = baseline_stats['avg']

    # 性能测试
    mean_latency, std_latency = model_latency_test(model, device, dataloader, opt.testtime)
    fps = opt.batch / mean_latency


    power_stats = measure_inference_power(model, device, dataloader, power_monitor, baseline_power,
                                          opt.power_iterations)
    efficiency = calculate_efficiency(fps, power_stats['avg'], baseline_power)


    logging.info("\n" + "=" * 50)
    logging.info(f"Weights: {opt.weights} ({get_weight_size(opt.weights)}MB)")
    logging.info(f"Device: {device} | Batch: {opt.batch} | Imgs: {opt.imgs}")
    logging.info(f"Latency: {mean_latency:.5f}s ± {std_latency:.5f}s")
    logging.info(f"FPS: {fps:.1f}")
    logging.info(f"Baseline power: {baseline_power:.2f}W")
    logging.info(f"Inference power: {power_stats['avg']:.2f}W (Δ {power_stats['avg'] - baseline_power:.2f}W)")
    logging.info(f"Efficiency: {efficiency:.2f} FPS/W")
    logging.info("=" * 50)


    if opt.test_batches:
        batch_results = different_batch_sizes_test(
            model, device, dataloader, power_monitor, baseline_power,
            batch_sizes=[1, 2, 4, 8, 16]
        )


    if opt.save_results:
        result_file = f"benchmark_{Path(opt.weights).stem}.txt"
        with open(result_file, 'w') as f:
            f.write(f"Test Report - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model: {opt.weights}\n")
            f.write(f"Dataset: {opt.data_dir} ({len(dataset)} images)\n")
            f.write(f"Baseline Power: {baseline_power:.2f}W\n\n")

            f.write("[Single Batch Results]\n")
            f.write(f"Batch Size: {opt.batch}\n")
            f.write(f"Latency: {mean_latency:.5f}s ± {std_latency:.5f}s\n")
            f.write(f"FPS: {fps:.1f}\n")
            f.write(f"Total Power: {power_stats['avg']:.2f}W\n")
            f.write(f"Delta Power: {power_stats['avg'] - baseline_power:.2f}W\n")
            f.write(f"Efficiency: {efficiency:.2f} FPS/W\n\n")

            if opt.test_batches:
                f.write("[Multi-Batch Results]\n")
                for res in batch_results:
                    f.write(f"Batch {res['batch_size']}:\n")
                    f.write(f"  Latency: {res['latency']:.5f}s ± {res['latency_std']:.5f}s\n")
                    f.write(f"  FPS: {res['fps']:.1f}\n")
                    f.write(f"  Total Power: {res['total_power']:.2f}W\n")
                    f.write(f"  Delta Power: {res['delta_power']:.2f}W\n")
                    f.write(f"  Efficiency: {res['efficiency']:.2f} FPS/W\n\n")

        logging.info(f"Results saved to {result_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 Real-Image Benchmark Tool')
    parser.add_argument('--weights', type=str, default=r"D:\DL_Project\YOLOv8-multi-task-main\runs\20250406-n-v7\weights\best.pt", help='Model weights path')
    parser.add_argument('--data-dir', type=str, default=r"D:\navigationData\ImagesForYOLOv8Mutil_3\images\val", help='Test images directory')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='Image size [h w]')
    parser.add_argument('--device', default='0', help='CUDA device (0) or cpu')
    parser.add_argument('--workers', type=int, default=4, help='Data loading workers')
    parser.add_argument('--warmup', type=int, default=200, help='Warmup iterations')
    parser.add_argument('--testtime', type=int, default=1000, help='Test iterations')
    parser.add_argument('--baseline-duration', type=int, default=30, help='Baseline measurement seconds')
    parser.add_argument('--power-iterations', type=int, default=100, help='Power test iterations')
    parser.add_argument('--half', action='store_true', help='Use FP16 precision')
    parser.add_argument('--test-batches', action='store_true', help='Test multiple batch sizes')
    parser.add_argument('--save-results', action='store_true', help='Save results to file')

    opt = parser.parse_args()
    main(opt)