import os

os.system("mkdir train_logs")
os.system("mkdir test_logs")
os.system("mkdir small_region")
for i in range(1, 13):
    os.system("mkdir new_model%d" % (i))
    os.system("mkdir small_model_mon%d" % (i))
    os.system("mkdir small_map_data_%d" % (i))

