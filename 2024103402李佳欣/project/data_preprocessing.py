import pandas as pd
import numpy as np
import jieba
import re
import os

class DataProcessor:
    def __init__(self, base_path='./ylww/'):
        self.base_path = base_path

    def load_original_data(self, file_name='评语.xlsx'):
        file_path = os.path.join(self.base_path, file_name)
        df = pd.read_excel(file_path)
        print(f"原始数据形状: {df.shape}")
        return df

    def merge_comment_columns(self, df):
        df['评语'] = df['评语'].fillna('')
        df['不足与建议'] = df['不足与建议'].fillna('')
        df['整体评语'] = df['评语'] + df['不足与建议']
        df = df.drop(['评语', '不足与建议'], axis=1)
        print(f"合并评语后数据形状: {df.shape}")
        return df

    def split_sentences(self, text):
        if not isinstance(text, str):
            return []
        number_pattern = r'\d+\.|\d+、|\(\d+\)|\d+\)|【\d+】|\[\d+\]|\d+，|一、|二、|三、|四、|五、|六、|七、|八、|九、|十、'
        sentences = re.split(r'(?<=[。！？；!?;])', text)
        merged_sentences = []
        buffer = ""
        has_number = False
        for sentence in sentences:
            if re.match(number_pattern, sentence.strip()):
                has_number = True
                if buffer:
                    merged_sentences.append(buffer.strip())
                    buffer = ""
                buffer += sentence
            else:
                if has_number:
                    buffer += sentence
                else:
                    merged_sentences.append(sentence.strip())
        if buffer:
            merged_sentences.append(buffer.strip())
        final_sentences = []
        for s in merged_sentences:
            if s:
                cleaned = re.sub(r'[^\w]', '', s).replace(" ", "")
                if cleaned:  # 只保留非空字符串
                    final_sentences.append(cleaned)
        return final_sentences

    def apply_sentence_splitting(self, df):
        df['整体评语_分句'] = df['整体评语'].apply(self.split_sentences)
        print(f"分句处理后数据形状: {df.shape}")
        return df

    def remove_unnecessary_columns(self, df):
        columns_to_drop = [
             '名称', '领域码','类型'
        ]
        df = df.drop(columns_to_drop, axis=1)
        print(f"删除不必要列后数据形状: {df.shape}")
        return df

    def expand_sentences(self, df):
        df['整体评语_分句'] = df['整体评语_分句'].apply(eval)
        df_expanded = df.explode('整体评语_分句')
        print(f"展开句子后数据形状: {df_expanded.shape}")
        return df_expanded

    def clean_expanded_data(self, df):
        df = df.dropna(subset=['整体评语_分句'])
        print(f"清理空值后数据形状: {df.shape}")
        return df

    def sample_data(self, df, sample_ratio=0.15, random_state=42):
        sample_size = int(len(df) * sample_ratio)
        sampled_df = df.sample(n=sample_size, random_state=random_state)
        print(f"抽样后数据形状: {sampled_df.shape}")
        return sampled_df

    def save_data(self, df, file_name):
        file_path = os.path.join(self.base_path, file_name)
        df.to_excel(file_path, index=False)
        print(f"数据已保存到: {file_path}")

    def deduplicate_text_file(self, input_file='segmentations.txt', output_file='segmentation.txt'):
        input_path = os.path.join(self.base_path, input_file)
        output_path = os.path.join(self.base_path, output_file)
        with open(input_path, "r", encoding="utf-8") as file:
            words = set(file.read().splitlines())
        with open(output_path, "w", encoding="utf-8") as file:
            for word in sorted(words):
                file.write(word + "\n")
        print(f"去重完成，结果已保存到 {output_path}")
        print(f"原始词汇数量: {len(set(open(input_path, 'r', encoding='utf-8').read().splitlines()))}")
        print(f"去重后词汇数量: {len(words)}")

def main():
    processor = DataProcessor()
    print("=== 开始数据处理流程 ===")

    print("\n1. 加载原始数据...")
    df = processor.load_original_data()

    print("\n2. 合并评语列...")
    df = processor.merge_comment_columns(df)
    processor.save_data(df, 'Merge.xlsx')

    print("\n3. 应用分句处理...")
    df = processor.apply_sentence_splitting(df)

    print("\n4. 删除不必要的列...")
    df = processor.remove_unnecessary_columns(df)
    processor.save_data(df, 'Merge1.xlsx')

    print("\n5. 展开句子...")
    df = processor.expand_sentences(df)
    processor.save_data(df, 'expanded.xlsx')

    print("\n6. 清理展开后的数据...")
    df = processor.clean_expanded_data(df)
    processor.save_data(df, 'expanded.xlsx')

    print("\n7. 随机抽样...")
    sampled_df = processor.sample_data(df)
    processor.save_data(sampled_df, 'Manual.xlsx')

    print("\n8. 文本文件去重...")
    processor.deduplicate_text_file()

    print("\n=== 数据处理流程完成 ===")

if __name__ == "__main__":
    main()