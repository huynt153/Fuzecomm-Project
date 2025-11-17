"""
Symbol Design Module with Gray Mapping and (12,8) Hamming Code
- 8 bits real information + 4 bits Hamming parity = 12 bits
- 4 bits CRC/Parity for error detection
- Preamble for synchronization
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple

@dataclass
class SymbolDesign:
    """
    Symbol Design Configuration:
    - Preamble: 8 bits (0xAA = 10101010 for sync)
    - Payload: 12 bits (8 info + 4 Hamming)
    - CRC: 4 bits for error detection
    Total: 24 bits = 3 bytes per symbol
    """
    
    PREAMBLE_SIZE = 8  # bits
    INFO_BITS = 8      # real information bits
    HAMMING_PARITY = 4 # Hamming parity bits
    PAYLOAD_BITS = 12  # INFO_BITS + HAMMING_PARITY
    CRC_BITS = 4       # CRC bits
    TOTAL_BITS = PREAMBLE_SIZE + PAYLOAD_BITS + CRC_BITS  # 24 bits = 3 bytes
    
    # Preamble pattern for synchronization (0xAA)
    PREAMBLE_PATTERN = 0b10101010
    
    # Gray code mapping for 3-bit symbols (0-7)
    GRAY_CODE_MAP = {
        0: 0b000,  # 0
        1: 0b001,  # 1
        2: 0b011,  # 3
        3: 0b010,  # 2
        4: 0b110,  # 6
        5: 0b111,  # 7
        6: 0b101,  # 5
        7: 0b100,  # 4
    }
    
    # Inverse Gray code mapping
    INVERSE_GRAY_CODE_MAP = {v: k for k, v in GRAY_CODE_MAP.items()}
    
    @staticmethod
    def binary_to_gray(n: int) -> int:
        """Convert binary to Gray code"""
        return n ^ (n >> 1)
    
    @staticmethod
    def gray_to_binary(n: int) -> int:
        """Convert Gray code to binary"""
        mask = n
        while mask:
            mask >>= 1
            n ^= mask
        return n
    
    @staticmethod
    def calculate_hamming_parity(data: int) -> int:
        """
        Calculate (12,8) Hamming code parity bits
        Data: 8 bits (d0-d7)
        Parity: 4 bits (p0-p3)
        Position: p0 p1 d0 p2 d1 d2 d3 p3 d4 d5 d6 d7
        """
        d = [(data >> i) & 1 for i in range(8)]
        
        # Calculate parity bits
        p0 = d[0] ^ d[1] ^ d[3] ^ d[4] ^ d[6]  # positions 3,5,7,9,11
        p1 = d[0] ^ d[2] ^ d[3] ^ d[5] ^ d[6]  # positions 3,6,7,10,11
        p2 = d[1] ^ d[2] ^ d[3] ^ d[7]         # positions 5,6,7,12
        p3 = d[4] ^ d[5] ^ d[6] ^ d[7]         # positions 9,10,11,12
        
        return (p3 << 3) | (p2 << 2) | (p1 << 1) | p0
    
    @staticmethod
    def encode_hamming(data: int) -> int:
        """
        Encode 8-bit data with (12,8) Hamming code
        Returns 12-bit codeword
        """
        parity = SymbolDesign.calculate_hamming_parity(data)
        # Interleave data and parity bits
        codeword = 0
        d_idx = 0
        p_idx = 0
        
        for i in range(12):
            pos = i + 1
            # Power of 2 positions are parity bits
            if pos & (pos - 1) == 0:  # Check if power of 2
                codeword |= ((parity >> p_idx) & 1) << i
                p_idx += 1
            else:
                codeword |= ((data >> d_idx) & 1) << i
                d_idx += 1
        
        return codeword
    
    @staticmethod
    def decode_hamming(codeword: int) -> Tuple[int, bool]:
        """
        Decode 12-bit Hamming codeword
        Returns: (data, error_corrected)
        """
        # Calculate syndrome
        syndrome = 0
        for i in range(4):
            parity_pos = (1 << i) - 1
            parity = 0
            for j in range(12):
                if ((j + 1) & (1 << i)) != 0:
                    parity ^= (codeword >> j) & 1
            syndrome |= parity << i
        
        error_corrected = False
        if syndrome != 0:
            # Single bit error detected, correct it
            error_pos = syndrome - 1
            codeword ^= (1 << error_pos)
            error_corrected = True
        
        # Extract data bits
        data = 0
        d_idx = 0
        for i in range(12):
            pos = i + 1
            if pos & (pos - 1) != 0:  # Not a power of 2
                data |= ((codeword >> i) & 1) << d_idx
                d_idx += 1
        
        return data, error_corrected
    
    @staticmethod
    def calculate_crc4(data: int, num_bits: int) -> int:
        """
        Calculate 4-bit CRC
        Polynomial: x^4 + x + 1 (0x13)
        """
        polynomial = 0b10011
        crc = 0
        data = data << 4  # Append 4 zeros
        
        for i in range(num_bits + 4 - 1, -1, -1):
            if ((crc ^ (data >> i)) & 1):
                crc = (crc << 1) ^ polynomial
            else:
                crc = crc << 1
            crc &= 0xF  # Keep only 4 bits
        
        return crc
    
    def encode_symbol(self, class_id: int) -> bytes:
        """
        Encode class_id (0-7) into full symbol with Gray mapping, Hamming, and CRC
        Returns 3 bytes
        """
        # Apply Gray mapping
        gray_code = self.binary_to_gray(class_id)
        
        # Pad to 8 bits for Hamming encoding
        data_8bit = gray_code & 0xFF
        
        # Apply (12,8) Hamming code
        hamming_encoded = self.encode_hamming(data_8bit)
        
        # Calculate CRC
        crc = self.calculate_crc4(hamming_encoded, self.PAYLOAD_BITS)
        
        # Construct full symbol: [Preamble 8b][Hamming 12b][CRC 4b] = 24 bits
        symbol = (self.PREAMBLE_PATTERN << 16) | (hamming_encoded << 4) | crc
        
        # Convert to 3 bytes
        return symbol.to_bytes(3, byteorder='big')
    
    def decode_symbol(self, symbol_bytes: bytes) -> Tuple[int, dict]:
        """
        Decode 3-byte symbol back to class_id
        Returns: (class_id, status_dict)
        """
        symbol = int.from_bytes(symbol_bytes, byteorder='big')
        
        # Extract components
        preamble = (symbol >> 16) & 0xFF
        hamming_encoded = (symbol >> 4) & 0xFFF
        received_crc = symbol & 0xF
        
        status = {
            'preamble_valid': preamble == self.PREAMBLE_PATTERN,
            'crc_valid': False,
            'hamming_corrected': False
        }
        
        # Check preamble
        if not status['preamble_valid']:
            return -1, status
        
        # Verify CRC
        calculated_crc = self.calculate_crc4(hamming_encoded, self.PAYLOAD_BITS)
        status['crc_valid'] = (calculated_crc == received_crc)
        
        if not status['crc_valid']:
            return -1, status
        
        # Decode Hamming
        data_8bit, error_corrected = self.decode_hamming(hamming_encoded)
        status['hamming_corrected'] = error_corrected
        
        # Reverse Gray mapping
        class_id = self.gray_to_binary(data_8bit & 0x7)
        
        return class_id, status
    
    def encode_batch(self, class_ids: np.ndarray) -> bytes:
        """Encode multiple class IDs"""
        encoded = b''
        for cid in class_ids:
            encoded += self.encode_symbol(int(cid))
        return encoded
    
    def decode_batch(self, encoded_data: bytes) -> Tuple[np.ndarray, List[dict]]:
        """Decode multiple symbols"""
        n_symbols = len(encoded_data) // 3
        class_ids = []
        statuses = []
        
        for i in range(n_symbols):
            symbol_bytes = encoded_data[i*3:(i+1)*3]
            cid, status = self.decode_symbol(symbol_bytes)
            class_ids.append(cid)
            statuses.append(status)
        
        return np.array(class_ids), statuses
    
    @classmethod
    def get_overhead_info(cls) -> dict:
        """Return detailed overhead information"""
        return {
            'preamble_bits': cls.PREAMBLE_SIZE,
            'information_bits': cls.INFO_BITS,
            'hamming_parity_bits': cls.HAMMING_PARITY,
            'crc_bits': cls.CRC_BITS,
            'total_bits_per_symbol': cls.TOTAL_BITS,
            'bytes_per_symbol': cls.TOTAL_BITS // 8,
            'overhead_bits': cls.PREAMBLE_SIZE + cls.HAMMING_PARITY + cls.CRC_BITS,
            'overhead_percentage': (cls.PREAMBLE_SIZE + cls.HAMMING_PARITY + cls.CRC_BITS) / cls.TOTAL_BITS * 100
        }"""
Symbol Design Module with Gray Mapping and (12,8) Hamming Code
- 8 bits real information + 4 bits Hamming parity = 12 bits
- 4 bits CRC/Parity for error detection
- Preamble for synchronization
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple

@dataclass
class SymbolDesign:
    """
    Symbol Design Configuration:
    - Preamble: 8 bits (0xAA = 10101010 for sync)
    - Payload: 12 bits (8 info + 4 Hamming)
    - CRC: 4 bits for error detection
    Total: 24 bits = 3 bytes per symbol
    """
    
    PREAMBLE_SIZE = 8  # bits
    INFO_BITS = 8      # real information bits
    HAMMING_PARITY = 4 # Hamming parity bits
    PAYLOAD_BITS = 12  # INFO_BITS + HAMMING_PARITY
    CRC_BITS = 4       # CRC bits
    TOTAL_BITS = PREAMBLE_SIZE + PAYLOAD_BITS + CRC_BITS  # 24 bits = 3 bytes
    
    # Preamble pattern for synchronization (0xAA)
    PREAMBLE_PATTERN = 0b10101010
    
    # Gray code mapping for 3-bit symbols (0-7)
    GRAY_CODE_MAP = {
        0: 0b000,  # 0
        1: 0b001,  # 1
        2: 0b011,  # 3
        3: 0b010,  # 2
        4: 0b110,  # 6
        5: 0b111,  # 7
        6: 0b101,  # 5
        7: 0b100,  # 4
    }
    
    # Inverse Gray code mapping
    INVERSE_GRAY_CODE_MAP = {v: k for k, v in GRAY_CODE_MAP.items()}
    
    @staticmethod
    def binary_to_gray(n: int) -> int:
        """Convert binary to Gray code"""
        return n ^ (n >> 1)
    
    @staticmethod
    def gray_to_binary(n: int) -> int:
        """Convert Gray code to binary"""
        mask = n
        while mask:
            mask >>= 1
            n ^= mask
        return n
    
    @staticmethod
    def calculate_hamming_parity(data: int) -> int:
        """
        Calculate (12,8) Hamming code parity bits
        Data: 8 bits (d0-d7)
        Parity: 4 bits (p0-p3)
        Position: p0 p1 d0 p2 d1 d2 d3 p3 d4 d5 d6 d7
        """
        d = [(data >> i) & 1 for i in range(8)]
        
        # Calculate parity bits
        p0 = d[0] ^ d[1] ^ d[3] ^ d[4] ^ d[6]  # positions 3,5,7,9,11
        p1 = d[0] ^ d[2] ^ d[3] ^ d[5] ^ d[6]  # positions 3,6,7,10,11
        p2 = d[1] ^ d[2] ^ d[3] ^ d[7]         # positions 5,6,7,12
        p3 = d[4] ^ d[5] ^ d[6] ^ d[7]         # positions 9,10,11,12
        
        return (p3 << 3) | (p2 << 2) | (p1 << 1) | p0
    
    @staticmethod
    def encode_hamming(data: int) -> int:
        """
        Encode 8-bit data with (12,8) Hamming code
        Returns 12-bit codeword
        """
        parity = SymbolDesign.calculate_hamming_parity(data)
        # Interleave data and parity bits
        codeword = 0
        d_idx = 0
        p_idx = 0
        
        for i in range(12):
            pos = i + 1
            # Power of 2 positions are parity bits
            if pos & (pos - 1) == 0:  # Check if power of 2
                codeword |= ((parity >> p_idx) & 1) << i
                p_idx += 1
            else:
                codeword |= ((data >> d_idx) & 1) << i
                d_idx += 1
        
        return codeword
    
    @staticmethod
    def decode_hamming(codeword: int) -> Tuple[int, bool]:
        """
        Decode 12-bit Hamming codeword
        Returns: (data, error_corrected)
        """
        # Calculate syndrome
        syndrome = 0
        for i in range(4):
            parity_pos = (1 << i) - 1
            parity = 0
            for j in range(12):
                if ((j + 1) & (1 << i)) != 0:
                    parity ^= (codeword >> j) & 1
            syndrome |= parity << i
        
        error_corrected = False
        if syndrome != 0:
            # Single bit error detected, correct it
            error_pos = syndrome - 1
            codeword ^= (1 << error_pos)
            error_corrected = True
        
        # Extract data bits
        data = 0
        d_idx = 0
        for i in range(12):
            pos = i + 1
            if pos & (pos - 1) != 0:  # Not a power of 2
                data |= ((codeword >> i) & 1) << d_idx
                d_idx += 1
        
        return data, error_corrected
    
    @staticmethod
    def calculate_crc4(data: int, num_bits: int) -> int:
        """
        Calculate 4-bit CRC
        Polynomial: x^4 + x + 1 (0x13)
        """
        polynomial = 0b10011
        crc = 0
        data = data << 4  # Append 4 zeros
        
        for i in range(num_bits + 4 - 1, -1, -1):
            if ((crc ^ (data >> i)) & 1):
                crc = (crc << 1) ^ polynomial
            else:
                crc = crc << 1
            crc &= 0xF  # Keep only 4 bits
        
        return crc
    
    def encode_symbol(self, class_id: int) -> bytes:
        """
        Encode class_id (0-7) into full symbol with Gray mapping, Hamming, and CRC
        Returns 3 bytes
        """
        # Apply Gray mapping
        gray_code = self.binary_to_gray(class_id)
        
        # Pad to 8 bits for Hamming encoding
        data_8bit = gray_code & 0xFF
        
        # Apply (12,8) Hamming code
        hamming_encoded = self.encode_hamming(data_8bit)
        
        # Calculate CRC
        crc = self.calculate_crc4(hamming_encoded, self.PAYLOAD_BITS)
        
        # Construct full symbol: [Preamble 8b][Hamming 12b][CRC 4b] = 24 bits
        symbol = (self.PREAMBLE_PATTERN << 16) | (hamming_encoded << 4) | crc
        
        # Convert to 3 bytes
        return symbol.to_bytes(3, byteorder='big')
    
    def decode_symbol(self, symbol_bytes: bytes) -> Tuple[int, dict]:
        """
        Decode 3-byte symbol back to class_id
        Returns: (class_id, status_dict)
        """
        symbol = int.from_bytes(symbol_bytes, byteorder='big')
        
        # Extract components
        preamble = (symbol >> 16) & 0xFF
        hamming_encoded = (symbol >> 4) & 0xFFF
        received_crc = symbol & 0xF
        
        status = {
            'preamble_valid': preamble == self.PREAMBLE_PATTERN,
            'crc_valid': False,
            'hamming_corrected': False
        }
        
        # Check preamble
        if not status['preamble_valid']:
            return -1, status
        
        # Verify CRC
        calculated_crc = self.calculate_crc4(hamming_encoded, self.PAYLOAD_BITS)
        status['crc_valid'] = (calculated_crc == received_crc)
        
        if not status['crc_valid']:
            return -1, status
        
        # Decode Hamming
        data_8bit, error_corrected = self.decode_hamming(hamming_encoded)
        status['hamming_corrected'] = error_corrected
        
        # Reverse Gray mapping
        class_id = self.gray_to_binary(data_8bit & 0x7)
        
        return class_id, status
    
    def encode_batch(self, class_ids: np.ndarray) -> bytes:
        """Encode multiple class IDs"""
        encoded = b''
        for cid in class_ids:
            encoded += self.encode_symbol(int(cid))
        return encoded
    
    def decode_batch(self, encoded_data: bytes) -> Tuple[np.ndarray, List[dict]]:
        """Decode multiple symbols"""
        n_symbols = len(encoded_data) // 3
        class_ids = []
        statuses = []
        
        for i in range(n_symbols):
            symbol_bytes = encoded_data[i*3:(i+1)*3]
            cid, status = self.decode_symbol(symbol_bytes)
            class_ids.append(cid)
            statuses.append(status)
        
        return np.array(class_ids), statuses
    
    @classmethod
    def get_overhead_info(cls) -> dict:
        """Return detailed overhead information"""
        return {
            'preamble_bits': cls.PREAMBLE_SIZE,
            'information_bits': cls.INFO_BITS,
            'hamming_parity_bits': cls.HAMMING_PARITY,
            'crc_bits': cls.CRC_BITS,
            'total_bits_per_symbol': cls.TOTAL_BITS,
            'bytes_per_symbol': cls.TOTAL_BITS // 8,
            'overhead_bits': cls.PREAMBLE_SIZE + cls.HAMMING_PARITY + cls.CRC_BITS,
            'overhead_percentage': (cls.PREAMBLE_SIZE + cls.HAMMING_PARITY + cls.CRC_BITS) / cls.TOTAL_BITS * 100
        }
