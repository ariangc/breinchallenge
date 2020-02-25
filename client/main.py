import json, requests, csv
import sys, os, argparse
import base64

import progressbar
from time import sleep

url = "http://54.173.172.217:9997/api/predict/"

def allowed_filename(filename):
	extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
	for e in extensions:
		if filename.endswith(e):
			return True, filename[:-len(e)] 
	
	return False, None

def main(data_path):
	image_ids = []
	labels = []
	total = len(os.listdir(data_path))
	bar = progressbar.ProgressBar(maxval=total,
			widgets=[progressbar.Bar('=','[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	print('Found {} images, starting predictions...'.format(str(total)))
	cont = 1
	for filename in os.listdir(data_path):
		cond, image_id = allowed_filename(filename)
		if cond:
			img = open(os.path.join(data_path, filename), 'rb').read()
			data = base64.b64encode(img)
			response = requests.post(url, data = data, headers = {'content-type': 'application/image'})
			data = json.loads(response.content)
			data_dict = json.loads(data)
			image_ids.append(image_id)
			labels.append(data_dict["class"] if data_dict["class"] != "jamon" else "mermelada")
			bar.update(cont)
			cont += 1
			sleep(0.1)

	with open("results.csv", 'w') as file:
		writer = csv.writer(file)
		writer.writerow(["image_id", "label"])
		for i in range(len(image_ids)):
			writer.writerow([image_ids[i], labels[i]])
	
	print("Succesfully predicted {} images! Check output in {}".format(str(len(image_ids)), 
			"results.csv"))

def parse_args():
	parser = argparse.ArgumentParser(description="Dataset divider")
	parser.add_argument("--data_path", required=True,
		help="Path to data")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args()
	main(args.data_path)

