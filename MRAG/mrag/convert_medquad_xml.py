import os
import glob
import json
import xml.etree.ElementTree as ET
import argparse

def parse_xml(file_path):
    qa_pairs = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        for qa in root.findall(".//QAPair"):
            q = qa.find("Question")
            a = qa.find("Answer")

            if q is not None and a is not None:
                question = q.text.strip() if q.text else ""
                answer = a.text.strip() if a.text else ""

                if question and answer:
                    qa_pairs.append({
                        "question": question,
                        "answer": answer
                    })
    except Exception as e:
        print(f"⚠️ Error parsing {file_path}: {e}")

    return qa_pairs

def scan_and_write(root_dir, out_file):
    scanned = 0
    all_qa_pairs = []

    # Recursively walk through all subfolders
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.xml'):
                scanned += 1
                xml_file = os.path.join(dirpath, filename)
                qa_pairs = parse_xml(xml_file)
                all_qa_pairs.extend(qa_pairs)

    if scanned == 0:
        raise RuntimeError(
            f"❌ No XML files found — checked directory and subdirectories: {root_dir}"
        )

    if not all_qa_pairs:
        raise RuntimeError(
            f"❌ No QA pairs extracted — checked {scanned} XML files. XML schema may differ: {root_dir}"
        )

    # Write all QA pairs to JSON
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Done! Extracted {len(all_qa_pairs)} QA pairs from {scanned} XML files.")

def _cli():
    parser = argparse.ArgumentParser(description="Convert MedQuAD XMLs to JSON")
    parser.add_argument("--root", required=True, help="Root directory of MedQuAD XML files")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    args = parser.parse_args()
    
    scan_and_write(args.root, args.out)

if __name__ == "__main__":
    _cli()
