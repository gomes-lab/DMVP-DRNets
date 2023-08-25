rm -r model
rm -r summary

for dataset in ebird
do
    for num in 1 2 3 4
    do 
        echo time python train.py $dataset $num
        python train.py $dataset $num
    done
done


