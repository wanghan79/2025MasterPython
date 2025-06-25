import argparse
import matplotlib.pyplot as plt
import pandas as pd
parser = argparse.ArgumentParser(description="")
parser.add_argument('--data',default='',type=str)
parser.add_argument('--compare_field',default='Test_Accuracy',type=str)
parser.add_argument('--grid',default=False,type=bool)
parser.add_argument('--title',default='',type=str)
parser.add_argument('--xlabel',default=None,type=str)
parser.add_argument('--ylabel',default=None,type=str)
args = parser.parse_args()

files = args.data
compare_key = args.compare_field.replace("_"," ")
for file in files.split(","):
    file_Name,name = file.split(":")
    df = pd.read_csv(file_Name)
    plt.plot(range(1,len(df[compare_key])+1),df[compare_key],label=name)
if args.xlabel is None:
    plt.xlabel('Epoch')
else:
    plt.xlabel(args.xlabel)
if args.ylabel is None:
    plt.ylabel(args.compare_field)
else:
    plt.ylabel(args.ylabel)
plt.title(args.title)
plt.legend()
plt.grid(args.grid)
plt.show()