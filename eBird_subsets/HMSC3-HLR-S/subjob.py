import sys
import os

exps = ["ebird"] #["birds", "butterfly", "plant", "trees", "vegetation"]
model = "train_NNGP"
path = "NNGP/"
queues = ["bigmem"] * 10 + ["inter"] * 10
cnt = 0
for exp in exps:
    for num in range(1,2):
        model_cp = model + "_%s_%s"%(exp, num)
        cmd1 = "#SBATCH -p %s -n 24 --mem=32G -J HSMC_GP -o %s.out -e %s.err --time=100:00:00 --exclusive "%\
        (queues[cnt], path + model_cp, path + model_cp) 
        cmd2 = "time Rscript %s.r %s %s"%(model, exp, num)
        print(cmd1)
        print(cmd2)
        print()
        f = open("tmp_run.sh",  "w")
        f.write("#!/bin/bash\n")
        f.write(cmd1 + "\n")
        f.write(cmd2 + "\n")
        f.close()
        os.system("sbatch tmp_run.sh")
        cnt += 1
