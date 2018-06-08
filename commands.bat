py prepare_data.py -d 'D:\Projects\Python\handwriting\Data\PointsLabelling\Texts' -c 'BORDERS' -o 'D:\Projects\Python\handwriting\Data\PointsLabelling\Prepared'
py training.py -i D:\Projects\Python\handwriting\Data\PointsLabelling\Prepared  -l 2 -u 512 -r 0.001 -o D:\Projects\Python\handwriting\Data\PointsLabelling\Trained3 -e 100
