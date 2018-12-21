cat model_1_* > model_w2v_1.h5
cat model_2_* > model_w2v_2.h5
cat model_3_* > model_w2v_3.h5
python3 test.py $1 $2 $3
