pwd

echo "fb_15_m2"
python3 main.py --save-dir "./fb_15_m2" --log-file "fb_15_m2_log.txt" --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 15

echo "fb_20_m2"
python3 main.py --save-dir "./fb_20_m2" --log-file "fb_20_m2_log.txt" --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 20 

echo "fb_30_m2"
python3 main.py --save-dir "./fb_30_m2" --log-file "fb_30_m2_log.txt" --K 6 --M 2 --T 6 --snr-ff -1 --snr-fb 30