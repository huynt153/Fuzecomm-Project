"""
Baseline 1: Conventional Communication
Raw sensor data → Δ-encoding → 8-bit quantization → Huffman → Compress to (2, 8, 12) bytes
ANS =>>>
- Conventional systems truyền raw data nên cần bandwidth lớn
- Không có semantic understanding → phải truyền tất cả thông tin
- Huffman chỉ giảm entropy nhưng không giảm được semantic redundancy
- FuzSemCom chỉ truyền "ý nghĩa" (8 classes) nên hiệu quả hơn
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import heapq
from collections import Counter
from typing import Tuple, Dict, List
import struct

@dataclass
class HuffmanNode:
    freq: int
    value: int = None
    left: 'HuffmanNode' = None
    right: 'HuffmanNode' = None
    
    def __lt__(self, other):
        return self.freq < other.freq

class ConventionalBaseline:
    """
    Conventional communication baseline with:
    1. Delta encoding (temporal correlation)
    2. 8-bit quantization
    3. Huffman coding
    4. Compression to fixed payload sizes: 2, 8, 12 bytes
    """
    
    SENSOR_FEATURES = ['Moisture', 'pH', 'N', 'Temperature', 'Humidity']
    
    # Quantization ranges (from domain knowledge)
    RANGES = {
        'Moisture': (0, 100),
        'pH': (4.0, 9.0),
        'N': (0, 200),
        'Temperature': (10, 40),
        'Humidity': (20, 100)
    }
    
    def __init__(self, target_bytes: int = 8):
        """
        target_bytes: 2, 8, or 12 bytes for comparison
        """
        self.target_bytes = target_bytes
        self.huffman_tree = None
        self.huffman_codes = {}
        self.prev_values = None
        
    def quantize_8bit(self, value: float, feature: str) -> int:
        """Quantize float value to 8-bit integer [0, 255]"""
        min_val, max_val = self.RANGES[feature]
        normalized = (value - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0, 1)
        return int(normalized * 255)
    
    def dequantize_8bit(self, quantized: int, feature: str) -> float:
        """Dequantize 8-bit integer back to float"""
        min_val, max_val = self.RANGES[feature]
        normalized = quantized / 255.0
        return min_val + normalized * (max_val - min_val)
    
    def delta_encode(self, values: np.ndarray) -> np.ndarray:
        """
        Delta encoding: store differences instead of absolute values
        First value is absolute, rest are deltas
        """
        if self.prev_values is None:
            # First sample: store absolute values
            self.prev_values = values.copy()
            return values
        
        # Compute deltas
        deltas = values - self.prev_values
        self.prev_values = values.copy()
        
        # Use 8-bit signed representation [-128, 127]
        deltas = np.clip(deltas, -128, 127).astype(np.int8)
        
        return deltas
    
    def build_huffman_tree(self, data: List[int]) -> HuffmanNode:
        """Build Huffman tree from data"""
        freq = Counter(data)
        heap = [HuffmanNode(f, v) for v, f in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)
        
        return heap[0] if heap else None
    
    def generate_huffman_codes(self, node: HuffmanNode, code: str = ''):
        """Generate Huffman codes from tree"""
        if node is None:
            return
        
        if node.value is not None:  # Leaf node
            self.huffman_codes[node.value] = code if code else '0'
            return
        
        self.generate_huffman_codes(node.left, code + '0')
        self.generate_huffman_codes(node.right, code + '1')
    
    def huffman_encode(self, data: List[int]) -> str:
        """Encode data using Huffman coding"""
        return ''.join(self.huffman_codes.get(val, '0') for val in data)
    
    def compress_to_target(self, bitstring: str) -> bytes:
        """
        Compress bitstring to target_bytes
        Uses truncation or padding
        """
        target_bits = self.target_bytes * 8
        
        if len(bitstring) > target_bits:
            # Truncate (lossy compression)
            bitstring = bitstring[:target_bits]
        else:
            # Pad with zeros
            bitstring = bitstring.ljust(target_bits, '0')
        
        # Convert to bytes
        byte_array = bytearray()
        for i in range(0, len(bitstring), 8):
            byte = bitstring[i:i+8]
            byte_array.append(int(byte, 2))
        
        return bytes(byte_array)
    
    def encode_sample(self, sample: Dict[str, float], is_first: bool = False) -> bytes:
        """
        Encode a single sensor sample
        Pipeline: Quantize → Delta encode → Huffman → Compress
        """
        # Step 1: Extract and quantize features
        quantized = np.array([
            self.quantize_8bit(sample[feat], feat) 
            for feat in self.SENSOR_FEATURES
        ])
        
        # Step 2: Delta encoding
        if is_first:
            self.prev_values = None
        encoded = self.delta_encode(quantized)
        
        # Step 3: Build Huffman tree (in practice, use pre-built tree)
        self.huffman_tree = self.build_huffman_tree(encoded.tolist())
        self.huffman_codes = {}
        self.generate_huffman_codes(self.huffman_tree)
        
        # Step 4: Huffman encode
        bitstring = self.huffman_encode(encoded.tolist())
        
        # Step 5: Compress to target bytes
        compressed = self.compress_to_target(bitstring)
        
        return compressed
    
    def encode_batch(self, samples: List[Dict[str, float]]) -> bytes:
        """Encode multiple samples"""
        encoded = b''
        for i, sample in enumerate(samples):
            encoded += self.encode_sample(sample, is_first=(i == 0))
        return encoded
    
    @staticmethod
    def calculate_compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
        """Calculate compression ratio"""
        return original_bytes / compressed_bytes
    
    @staticmethod
    def get_overhead_info(target_bytes: int) -> dict:
        """Return overhead information"""
        # Original: 5 features × 4 bytes (float32) = 20 bytes
        original_bytes = 5 * 4
        
        return {
            'original_size_bytes': original_bytes,
            'compressed_size_bytes': target_bytes,
            'compression_ratio': original_bytes / target_bytes,
            'bandwidth_reduction_percent': (1 - target_bytes / original_bytes) * 100,
            'encoding_steps': ['Delta-encoding', '8-bit quantization', 'Huffman coding', 'Truncation/Padding'],
            'lossy': target_bytes < original_bytes,
            'answer_why_compress': (
                "Conventional systems transmit raw sensor data without semantic understanding. "
                "Even with compression (Delta-encoding + Huffman), they still need significant bandwidth "
                "because they preserve all information. FuzSemCom only transmits semantic meaning "
                "(8 classes in 3 bytes) instead of raw values (20 bytes), achieving better efficiency."
            )
        }

class ConventionalEvaluator:
    """Evaluate Conventional Baseline performance"""
    
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.baseline_2B = ConventionalBaseline(target_bytes=2)
        self.baseline_8B = ConventionalBaseline(target_bytes=8)
        self.baseline_12B = ConventionalBaseline(target_bytes=12)
    
    def evaluate(self, df) -> dict:
        """Evaluate all three variants"""
        results = {}
        
        for name, baseline in [
            ('2byte', self.baseline_2B),
            ('8byte', self.baseline_8B),
            ('12byte', self.baseline_12B)
        ]:
            # Prepare samples
            samples = []
            for _, row in df.iterrows():
                samples.append({
                    'Moisture': row['Moisture'],
                    'pH': row['pH'],
                    'N': row['N'],
                    'Temperature': row['Temperature'],
                    'Humidity': row['Humidity']
                })
            
            # Encode
            encoded = baseline.encode_batch(samples)
            
            results[name] = {
                'total_bytes': len(encoded),
                'bytes_per_sample': len(encoded) / len(samples),
                'compression_ratio': 20 / (len(encoded) / len(samples)),
                'overhead_info': baseline.get_overhead_info(baseline.target_bytes)
            }
        
        return results
