import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

def check_fast(file_path, chunk_size=1000000, max_chunks=None):
    """
    快速检查非数值 - 使用向量化操作
    """
    print("="*80, flush=True)
    print("FAST Check for Non-Numeric Values", flush=True)
    print("="*80, flush=True)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}", flush=True)
        return
    
    file_size = os.path.getsize(file_path) / (1024**3)
    print(f"File size: {file_size:.2f} GB", flush=True)
    print(f"Chunk size: {chunk_size:,} rows", flush=True)
    if max_chunks:
        print(f"Max chunks to process: {max_chunks}", flush=True)
    print(flush=True)
    
    total_rows = 0
    total_original_nan = 0
    total_non_numeric = 0
    non_numeric_samples = {}  # 只保存前100个unique值作为样本
    
    chunk_num = 0
    start_time = datetime.now()
    
    print("Starting to read file...", flush=True)
    print("-"*80, flush=True)
    
    try:
        for chunk in pd.read_csv(
            file_path,
            usecols=['subject_id', 'valuenum', 'omop_concept_id'],
            chunksize=chunk_size,
            dtype={'subject_id': 'int32', 'omop_concept_id': 'str'},
            # 关键：让pandas自动推断valuenum的类型
        ):
            chunk_num += 1
            chunk_start = datetime.now()
            
            print(f"\nChunk {chunk_num}:", flush=True)
            chunk_rows = len(chunk)
            total_rows += chunk_rows
            
            # 统计原始NaN
            original_nan = chunk['valuenum'].isna().sum()
            total_original_nan += original_nan
            
            # 关键优化：使用pd.to_numeric的errors='coerce'来快速识别
            # 这比逐个apply快得多
            numeric_converted = pd.to_numeric(chunk['valuenum'], errors='coerce')
            
            # 找出哪些原本不是NaN，但转换后变成了NaN
            was_not_nan = chunk['valuenum'].notna()
            became_nan = numeric_converted.isna()
            non_numeric_mask = was_not_nan & became_nan
            
            chunk_non_numeric_count = non_numeric_mask.sum()
            total_non_numeric += chunk_non_numeric_count
            
            # 如果有非数值，收集样本
            if chunk_non_numeric_count > 0:
                non_numeric_values = chunk.loc[non_numeric_mask, 'valuenum']
                unique_vals = non_numeric_values.unique()
                
                # 只保存前100个unique值的样本
                for val in unique_vals[:100]:
                    if len(non_numeric_samples) < 100:
                        count = (non_numeric_values == val).sum()
                        if val in non_numeric_samples:
                            non_numeric_samples[val] += count
                        else:
                            non_numeric_samples[val] = count
            
            chunk_time = (datetime.now() - chunk_start).total_seconds()
            
            print(f"  Rows: {chunk_rows:,}", flush=True)
            print(f"  Original NaN: {original_nan:,}", flush=True)
            print(f"  Non-numeric: {chunk_non_numeric_count:,}", flush=True)
            print(f"  Time: {chunk_time:.1f}s", flush=True)
            print(f"  Total processed: {total_rows:,} rows", flush=True)
            
            # 每10个chunk显示汇总
            if chunk_num % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_rows / elapsed if elapsed > 0 else 0
                print(f"\n  === Progress Summary ===", flush=True)
                print(f"  Total rows so far: {total_rows:,}", flush=True)
                print(f"  Total non-numeric found: {total_non_numeric:,}", flush=True)
                print(f"  Processing rate: {rate:,.0f} rows/sec", flush=True)
                print(f"  Elapsed time: {elapsed/60:.1f} minutes", flush=True)
            
            # 如果设置了max_chunks限制
            if max_chunks and chunk_num >= max_chunks:
                print(f"\n⚠️  Reached max chunks limit ({max_chunks})", flush=True)
                break
                
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user at chunk {chunk_num}", flush=True)
    except Exception as e:
        print(f"\n❌ Error at chunk {chunk_num}:", flush=True)
        print(f"{e}", flush=True)
        import traceback
        traceback.print_exc()
    
    # 最终结果
    elapsed_total = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*80, flush=True)
    print("RESULTS", flush=True)
    print("="*80, flush=True)
    
    print(f"\nChunks processed: {chunk_num}", flush=True)
    print(f"Total rows: {total_rows:,}", flush=True)
    print(f"Processing time: {elapsed_total/60:.1f} minutes", flush=True)
    print(f"Average rate: {total_rows/elapsed_total:,.0f} rows/sec", flush=True)
    
    print(f"\nOriginal NaN: {total_original_nan:,} ({total_original_nan/total_rows*100:.2f}%)", flush=True)
    print(f"Non-numeric values: {total_non_numeric:,} ({total_non_numeric/total_rows*100:.2f}%)", flush=True)
    print(f"Total missing after conversion: {total_original_nan + total_non_numeric:,} "
          f"({(total_original_nan + total_non_numeric)/total_rows*100:.2f}%)", flush=True)
    
    if total_non_numeric == 0:
        print("\n✓ NO non-numeric values found!", flush=True)
        if max_chunks:
            print(f"  (Note: Only checked first {chunk_num} chunks)", flush=True)
        return
    
    print(f"\n⚠️  Found {total_non_numeric:,} non-numeric values!", flush=True)
    print(f"Unique non-numeric values (sample): {len(non_numeric_samples)}", flush=True)
    
    if len(non_numeric_samples) > 0:
        print("\n" + "-"*80, flush=True)
        print("Sample of Non-Numeric Values:", flush=True)
        print("-"*80, flush=True)
        
        sorted_samples = sorted(non_numeric_samples.items(), 
                               key=lambda x: x[1], reverse=True)
        
        for i, (val, count) in enumerate(sorted_samples[:30], 1):
            pct = count / total_rows * 100
            print(f"{i:2d}. '{val}' - {count:,} occurrences ({pct:.4f}%)", flush=True)
        
        # 保存样本到文件
        output_file = 'non_numeric_samples.txt'
        with open(output_file, 'w') as f:
            f.write("Non-Numeric Value Samples\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total rows checked: {total_rows:,}\n")
            f.write(f"Non-numeric found: {total_non_numeric:,}\n\n")
            f.write("Values (sorted by frequency):\n")
            f.write("-"*80 + "\n")
            for val, count in sorted_samples:
                f.write(f"'{val}': {count:,}\n")
        
        print(f"\n✓ Samples saved to: {output_file}", flush=True)
    
    print("\n" + "="*80, flush=True)
    print("Done!", flush=True)
    print("="*80, flush=True)


def main():
    print("="*80, flush=True)
    print("FAST CATEGORICAL VALUE CHECK", flush=True)
    print("="*80, flush=True)
    print(f"Start time: {datetime.now()}", flush=True)
    print(f"Python: {sys.version}", flush=True)
    print(f"Pandas: {pd.__version__}", flush=True)
    print(f"Working directory: {os.getcwd()}", flush=True)
    print(flush=True)
    
    file_path = 'mimic_data/d_labevent_with_loinc.csv'
    
    # 可以先测试前N个chunks
    # check_fast(file_path, chunk_size=1000000, max_chunks=10)  # 只处理前10个chunks测试
    
    # 或者处理整个文件
    check_fast(file_path, chunk_size=1000000, max_chunks=None)


if __name__ == "__main__":
    main()