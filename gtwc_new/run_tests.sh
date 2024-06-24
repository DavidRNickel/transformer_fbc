pwd

echo "basic_test"
python3 testing.py --save-dir "./basic_test" --log-file "test_results.txt" --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 15 --loadfile "./basic_test/20240611-182735.pt" --num-validation-epochs 10000