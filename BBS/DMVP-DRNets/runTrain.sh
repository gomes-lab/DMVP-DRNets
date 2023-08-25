rm -r model
rm -r summary

for dataset in BBS
do
    for num in 1 
    do 
        echo time python train.py $dataset $num
        python train.py $dataset $num
    done
done


