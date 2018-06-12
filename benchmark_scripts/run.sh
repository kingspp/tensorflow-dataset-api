#!/usr/bin/env bash


echo "Running EP1"
python3 -u EP1_Basic_Placeholders.py cpu > logs/ep1_cpu.log ;
python3 -u EP1_Basic_Placeholders.py gpu > logs/ep1_gpu.log ;

echo "Running EP3"
python3 -u EP3_Feedable_Dataset.py cpu > logs/ep3_cpu.log ;
python3 -u EP3_Feedable_Dataset.py gpu > logs/ep3_gpu.log ;

echo "Running EP5"
python3 -u EP5_Feedable_Iterator.py cpu > logs/ep5_cpu.log ;
python3 -u EP5_Feedable_Iterator.py gpu > logs/ep5_gpu.log ;

echo "Running EP11"
python3 -u EP11_ReInitializable_Iterator_Switch.py cpu > logs/ep11_cpu.log ;
python3 -u EP11_ReInitializable_Iterator_Switch.py gpu > logs/ep11_gpu.log ;

echo "Running EP14"
python3 -u EP14_Feedable_Iterator_Multiple_Dataset.py cpu > logs/ep14_cpu.log ;
python3 -u EP14_Feedable_Iterator_Multiple_Dataset.py gpu > logs/ep14_gpu.log ;

echo "Running EP17"
python3 -u EP17_Replace_Placeholder.py cpu > logs/ep17_cpu.log ;
python3 -u EP17_Replace_Placeholder.py gpu > logs/ep17_gpu.log ;
