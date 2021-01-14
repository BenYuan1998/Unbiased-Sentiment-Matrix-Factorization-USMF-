# Usage: python sort_reviews_by_user.py data-dir output-dir

import sys
import os
import json

def process_category(file, output_dir):
    user_dict = {}
    with open(file) as data_file:
        for line in data_file:
            json_data = json.loads(line)
            user_id = json_data["reviewerID"]
            if user_id not in user_dict:
                user_dict[user_id] = []
            user_dict[user_id].append(json_data)
        total_Review = 0
        for k, v in user_dict.items():
            if len(v) >= 50:
                total_Review += len(v)
                v = sorted(v, key=lambda k: k['unixReviewTime'], reverse=True)
                filename = os.path.join(output_dir, k + '.json')
                with open(filename, 'a') as file:
                    for json_record in v:
                        file.write(json.dumps(json_record) + '\n')
        print(total_Review)




if len(sys.argv) < 3:
    sys.exit('Usage: %s data-dir output-dir' % sys.argv[0])
output_path = sys.argv[2]
data_dir = sys.argv[1]



files = os.listdir(data_dir)
for file in files:
    if os.path.splitext(file)[1] == '.json':
        output_dir = os.path.join(output_path, os.path.splitext(file)[0])
        is_exist = os.path.exists(output_dir)
        if not is_exist:
            os.makedirs(output_dir)
        process_category(os.path.join(data_dir, file), output_dir)

