pwd

echo "fb_15"
python3 main.py --save-dir "./fb_15" --log-file "fb_15_log.txt" --K 6 --M 3 --T 9 --snr-ff -1 --snr-fb 15

echo "fb_20"
python3 main.py --save-dir "./fb_20" --log-file "fb_20_log.txt" --K 6 --M 3 --T 9 --snr-ff -1 --snr-fb 20 

echo "fb_30"
python3 main.py --save-dir "./fb_30" --log-file "fb_30_log.txt" --K 6 --M 3 --T 9 --snr-ff -1 --snr-fb 30