cat model_1_* > model_1.h5
cat model_2_* > model_2.h5
cat model_3_* > model_3.h5
cat model_4_* > model_4.h5
python3 test.py $1 $2
