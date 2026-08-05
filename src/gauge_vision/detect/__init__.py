"""Gösterge tespiti — karede göstergenin nerede olduğunu bulur (İP5).

    dataset.py   Etiket dönüşümleri ve karışık eğitim kümesinin kurulması

Zincirdeki yeri: tespit → kırpım → `read/needle.py` (İP6) → `read/calibrate.py` (İP7).
İP6'nın ölçümü tespitin verdiği **kutu merkezine** duyarlıdır (8 px kayma açı
hatasını 0,12°'den 3,65°'ye çıkarıyor), bu yüzden burada IoU tek başına yeterli
bir kabul ölçütü değildir.
"""
