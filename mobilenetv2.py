"""Download, train and run MobileNetV2 for the AM3d Study One comparison."""

import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1
from http.client import HTTPException
import json
from math import ceil
import os
from pathlib import Path
from statistics import mean, stdev
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import cv2
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras


REVISION = "52fe75dec70c1e8046493b4e3256f8f752a4d39d"
REPO = "mahsan2/AMdataset"
CLASSES = ("badcylinder", "goodcylinder")
DEFECT = 0
SIZE = 224
SEEDS = (42, 43, 44)
EPOCHS = 30
BATCH = 30
LR = 0.001
ARMS = ("clean", "augmented")
OPTIMIZERS = {"adam": keras.optimizers.Adam, "sgd": keras.optimizers.SGD,
              "rmsprop": keras.optimizers.RMSprop}
PAPER = {"adam": 1.0, "sgd": 54 / 55, "rmsprop": 1.0}
VIEWS = ("clean", "blur9_sigma2.0", "jpeg_quality15", "resize56", "rotate10_reflect")
CONDITIONS = ("clean", "brightness0.5", "brightness1.5", *VIEWS[1:])
RULE = ("Adopt augmentation if mean validation F1 improves by at least 0.01, "
        "clean validation F1 is maintained, and worst condition recall is maintained.")



# Git blob checksums from the pinned AMdataset tree for Dataset One only.
DATASET_BLOBS = """\
badcylinder/IMG_9078.JPG 05d963520a2a7e2eab5f6ab63791b7fb83dd7377
badcylinder/IMG_9079.JPG e8b91b80683a1b17697e85924f447a76ad7cd1ed
badcylinder/IMG_9080.JPG a62abdaeefa3b7ec6fed859f55991768d04f452b
badcylinder/IMG_9081.JPG b2a26cf1558eed5927e454c29a61879e4da8dfc5
badcylinder/IMG_9082.JPG a396fbd85aa60e222afbde7f1c36863e6655eefd
badcylinder/IMG_9083.JPG fee3a659244262c923343a18aeae9b4eab131817
badcylinder/IMG_9084.JPG fae3afaebf25495a40aab1c93a95d6e1a09609bd
badcylinder/IMG_9085.JPG 603674224ea4d276ecf371a7aaf1b42b09030b87
badcylinder/IMG_9086.JPG f5961aea724e5bce92b8a9c527ea5aee98383fea
badcylinder/IMG_9087.JPG 94ce0e9e61b9677d82e537146adbeb22cc74923a
badcylinder/IMG_9088.JPG cae19af771a8bcebf11c332356a553a0eab1ce1e
badcylinder/IMG_9089.JPG 8e792c3c0679a1e41cb917ac853ce2dea51fd91d
badcylinder/IMG_9090.JPG c15c46d191980df14fe4006c7e4c4f55e6142497
badcylinder/IMG_9091.JPG e0907d2af0017ca581bc2250a49b2f7dd03195d2
badcylinder/IMG_9092.JPG 3c064896bd7d938dbb71e4244336ab4f158d0dd0
badcylinder/IMG_9093.JPG dce991758f2bb6b98e2f216a36e54f1d42ae04b2
badcylinder/IMG_9094.JPG 29bcfa2d19f0df7a268557efbc5b5b950a5b2081
badcylinder/IMG_9095.JPG e6d8a47e492ada6e26f633f39cc44950a90a3b90
badcylinder/IMG_9096.JPG 6b2680538d5bc107671a712971e1f4f20da2a555
badcylinder/IMG_9097.JPG 5b3a4d4377cb3f52cf6a070e8b203488355cfc8b
badcylinder/IMG_9098.JPG 0b58bddcce53f80be595eba6cc9dc233bdcf1e76
badcylinder/IMG_9099.JPG cfdfe58274c87644b39870e6c1a7c446dbe42721
badcylinder/IMG_9100.JPG 81e71195394a6e62d42983697d4bf2aceb60ba44
badcylinder/IMG_9101.JPG cbb59df89aae91176678fb4531d02642ec7c1c91
badcylinder/IMG_9102.JPG f91c99a500f1cbc9683fe10d23ddc4ca06f2b7ff
badcylinder/IMG_9103.JPG 43618f659f50d2972598b1ea5281919e1ff13088
badcylinder/IMG_9104.JPG afc03515868f4d6ff192af276acdf7da74730cd9
badcylinder/IMG_9105.JPG f6db2b31508c3242ae852e5f476922617fc0ecd7
badcylinder/IMG_9106.JPG 6632697fca245dd9e8088231affe6c78bb4a768c
badcylinder/IMG_9107.JPG 2074836194d59d88c8fc1dceec3bd5371c634ab0
badcylinder/IMG_9108.JPG abd7aa0435396084e09ea9a36700fa66d2288a0f
badcylinder/IMG_9109.JPG a3bfa048a7dd171b2a1037216836304040cafed1
badcylinder/IMG_9110.JPG 0d3bb8e637f15200f959a0f7ec979a77852ad0f6
badcylinder/IMG_9111.JPG 22efb11b5b328e0b4e0a13a64f83319c16100676
badcylinder/IMG_9112.JPG e5d1dfa90f295c21801966fe92dc8a63a6767198
badcylinder/IMG_9113.JPG cc9ccee9dbdc50c32bb4bc03e51b99c3359356e6
badcylinder/IMG_9114.JPG d54eb6dbb2969bb7a96993b9a095d1e436fcdc35
badcylinder/IMG_9115.JPG 038ac581026b2813b4bf5a965eabb06517996ed4
badcylinder/IMG_9116.JPG 7d6000b1bbd73b3d5005cb06c8b761c7e5bd6936
badcylinder/IMG_9117.JPG cf881cd65f1ddb6b13ae6961de15e1a07cad1da1
badcylinder/IMG_9118.JPG 6f2f21046bb42cbb52ace8d696b9db4bbf84adfc
badcylinder/IMG_9119.JPG e9a8919fce2b397c3297d3c4dd3156bac1043c32
badcylinder/IMG_9120.JPG 3be627a0a3901a0798b73beb5feb6f46f06e27db
badcylinder/IMG_9121.JPG a50e4c7dde58c2565249d590bab21100a7dcc849
badcylinder/IMG_9122.JPG 12dbc935f8a42309a913e158c20350d9fcc3317c
badcylinder/IMG_9123.JPG 136c1d5219588643d39adcc18f24f4d0105c3ee7
badcylinder/IMG_9124.JPG 2f205fc2d768ac3e7b1d3a510d2f1b53a1d3eeca
badcylinder/IMG_9125.JPG e95771bffb67fa0086f2fa842c6d4f2898377296
badcylinder/IMG_9126.JPG ed7defa84884b11cff0cd54cf0b8c43d47f31b1a
badcylinder/IMG_9127.JPG ae6ce08192b26e37f26e5e9bddc958f160941838
badcylinder/IMG_9128.JPG 330f532ce466bbd45a69861775f8489074928f22
badcylinder/IMG_9129.JPG 64d8c5c2cd8210f35d8f0cd30527e11d3c6b628d
badcylinder/IMG_9130.JPG 4444ab0393406d46660df2db680792b284d13f9a
badcylinder/IMG_9131.JPG b4eb0001db6bf4b29532448e835d1d312e9123b9
badcylinder/IMG_9132.JPG a97edff1177b8b080ad56094c0fbee34e374273f
badcylinder/IMG_9133.JPG b2bfad822ef64d3852c0d05518e68526a4bd904c
badcylinder/IMG_9134.JPG 9f2fbb1dba7ab65955ad8901121cbeaff60efb5d
badcylinder/IMG_9135.JPG 7db6f86ca893f2c7558ad080a021fe093475fb0b
badcylinder/IMG_9136.JPG f435fd1c2bb0931215ea9532640690bf153dd10e
badcylinder/IMG_9137.JPG e8780cfccf93967d6d67f06cf55925fc59e2ce5a
badcylinder/IMG_9138.JPG 61bd87921d04112d2b5a746c722fd3b61ddd97a3
badcylinder/IMG_9139.JPG ee3c2d00fd060fcdd359e7cc83637eff31455196
badcylinder/IMG_9140.JPG 9ce3c10f99384545315ddc76a572bd02b3268a13
badcylinder/IMG_9141.JPG e042eeb47ef52967fb2711402c93c994396c65b1
badcylinder/IMG_9142.JPG e70d78233ffabe5d9a6b2892c04485c95405e135
badcylinder/IMG_9143.JPG 5130ca6e3f04688fe377b8e33750726789a71bab
badcylinder/IMG_9144.JPG ced5659ad5dca97ab296a365d164261162018507
badcylinder/IMG_9145.JPG ed19672ea73917d01c70feed1656584e32b1713a
badcylinder/IMG_9146.JPG d2567cce3be15759c423da0475fb309599af0159
badcylinder/IMG_9147.JPG 33ad4f425625d188925166192a0da6ff6a4dfe83
badcylinder/IMG_9148.JPG 53e3d67acef3208ecf1152c0f7c78e7d6ed294ff
badcylinder/IMG_9149.JPG 505e04e8b339091103fdb67201b4ccf19566446f
badcylinder/IMG_9150.JPG 1c58cfd36f5f1e039731722398fafa1b032ef492
badcylinder/IMG_9151.JPG 119eb3a839397d8d7b1a7cc9651ea2f3837d7ca2
badcylinder/IMG_9152.JPG b1feb6f783cc330fbcd90e02a65d19e70d832a58
badcylinder/IMG_9153.JPG a31239410331d0d305794361ad77246b6e78f64f
badcylinder/IMG_9154.JPG 822dd53ea9f1b03ebd3f19d4f2a162e50215f3b4
badcylinder/IMG_9155.JPG 1f880ea1eb38210893c1e42c3e06f15f37445cdb
badcylinder/IMG_9156.JPG 3be515763a5487d76a646d8a9d7cee4eaf13dd6d
badcylinder/IMG_9157.JPG 7ded2742e8e0041f72c182c4d8437c601624cece
badcylinder/IMG_9158.JPG f250b274fb377aecc199c6ba204ed4bcaa81702e
badcylinder/IMG_9159.JPG ff7abe0166f04d6535605dd88f1cacf8935d7db2
badcylinder/IMG_9160.JPG 8715625ed9a8236d914e8e1dbff945c01d7b9fb2
badcylinder/IMG_9161.JPG 1aaa77e174fe727e83987e07fbc9b190cfcc6337
badcylinder/IMG_9162.JPG dfda9742386df82836fea52e8d2e35e105050f32
badcylinder/IMG_9163.JPG baaccce79b87815273883f2b4907c675903d2f4c
badcylinder/IMG_9164.JPG a1f73ecfd0f1bcbd1d20dc049cbb6a5a97a1d5e7
badcylinder/IMG_9165.JPG 76df326a97baaa3ec33a67b70ec1ab982a64f9d7
badcylinder/IMG_9166.JPG 69b7a9cffa46c3d544a0eb39cdde04cfe5045f3b
badcylinder/IMG_9167.JPG cff74ad678e76c6086e27bc0a95915d2e82c5687
badcylinder/IMG_9168.JPG 93d63b75b9895fb2fac7a0a784b41bc3da0e18ed
badcylinder/IMG_9169.JPG 0a7f610acf5ce1987efa855ea95182daa12ecd02
badcylinder/IMG_9170.JPG 2f834b808de22b6a76cab8dda2f86868cb3ca491
badcylinder/IMG_9171.JPG 2ea42d3bc15edef2a7bfb0c0998255db8e9f5ae1
badcylinder/IMG_9172.JPG 089c5f697cefe80b001619916fe3028e4c60519b
badcylinder/IMG_9173.JPG e681f05adbd6ca2b01b4fcdfcd952aeeae0de4ff
badcylinder/IMG_9174.JPG 90be0b8d1e85073048723acc8ce2c194537e59cf
badcylinder/IMG_9175.JPG 382898f31096f4d69959224c56f25923922887c8
badcylinder/IMG_9176.JPG 7aa5a449ad9ab51693cc30beaa9ffc7a6269a885
badcylinder/IMG_9177.JPG fd4a7e8c86f3c62108543854baab900ec9769e76
badcylinder/IMG_9178.JPG c08b9e67642ec5ccdd673df3b2908fb9bb5a88c5
badcylinder/IMG_9179.JPG 9dacc9c74d001900c968930b1c023a213a686c91
badcylinder/IMG_9180.JPG 1ba9f8104e0b1ece94fadcdf163ccc5c8a0c4950
badcylinder/IMG_9181.JPG 3a0780ecce4fcc89e786ef66291cb05ef47810a0
badcylinder/IMG_9182.JPG c13e7939e29a126c4eebe4ed7c8185eba13bd648
badcylinder/IMG_9183.JPG 38353e92a9090c6f1f09f80c73b19b8a7bc3edfe
badcylinder/IMG_9184.JPG b71b5770fb8483a03ceb942b9eb333a2dfe2b4f2
badcylinder/IMG_9185.JPG b7510a6ae06c971b6bcb51cc146aac2f6413d5cd
badcylinder/IMG_9186.JPG d21891b7a1b88ce5a6cdf29c61d8ec28deb77a6f
badcylinder/IMG_9187.JPG e4607c80e0e689f10bd3c663e8e9ec965b38f5cf
badcylinder/IMG_9188.JPG 772e4179da49de163d42b4e9295bb30991a96e44
badcylinder/IMG_9189.JPG 94246f5395c2f6b699f94767817edc481d59731c
badcylinder/IMG_9190.JPG b73137b8d71533d00890704926ef67b6bd917bc9
badcylinder/IMG_9191.JPG 0ed1cc124da682db9c694a31db33b28f8e24517c
badcylinder/IMG_9192.JPG c636b5e9a02fd261e35b8d4e368ea6e4973b9c28
badcylinder/IMG_9193.JPG 8102ac6df78dcd81380a8ec91fc5d9906f549fd4
badcylinder/IMG_9194.JPG 74290eb0535b59dfaec2fc07ff3e782c174ba148
badcylinder/IMG_9195.JPG 15840369afe335e2dbf65834bfec0160a852bf76
badcylinder/IMG_9196.JPG 5178bd56431e2162ec85c47412ac0d55c760cf48
badcylinder/IMG_9197.JPG bcde14fe37cdf0c2ee7346ca057eec8aed05f016
badcylinder/IMG_9198.JPG 424a1cfce6bd3372e93cf04948e5319963dd33d2
badcylinder/IMG_9199.JPG aaac270ce8a4ff87859faf7e5eb339f5070623e0
badcylinder/IMG_9200.JPG be5014176512f7bf5bf7519963188a4e7588a886
badcylinder/IMG_9201.JPG e85f450d0d267f0346c2fbf9ff84fd005dc11f2b
badcylinder/IMG_9202.JPG b8ecdd08402e205e8a7d5856bfa5db8cb0db7a29
badcylinder/IMG_9203.JPG d2e4c180d9008240da11aad77e3a5d4cf7a459bf
badcylinder/IMG_9204.JPG e3ee945530de15a918107a4d3d3e0194cf47cc62
badcylinder/IMG_9205.JPG 9bbc013ab3d6a5fc43fe2598d9981f86bfc867e5
badcylinder/IMG_9206.JPG 1338c330140fce6c1105403dd6f2365f2d0db830
badcylinder/IMG_9207.JPG 640563159fd6527d572671bee1ed96cce9bae10f
badcylinder/IMG_9208.JPG d4bf2b2c450996520807dcbb59c573bd62f45766
badcylinder/IMG_9209.JPG 1e95b208ca3298961f95133d66a1a9d9c1d6fcdd
badcylinder/IMG_9210.JPG a0e4ffe60c9e526050cdf81a466bcb66094a0d09
badcylinder/IMG_9211.JPG 8dd81a86d9cb52f72b704c178855d224e8ec2047
badcylinder/IMG_9212.JPG 2a10dff7f96ba49ec1e89cea02849ef8686540c4
badcylinder/IMG_9213.JPG 4f8c08a4d6257efad1e1368d1620a1fb141330a9
goodcylinder/IMG_8915.JPG eba5f1e2488bef2daa423159feae26bcfb69ea06
goodcylinder/IMG_8916.JPG f88f4b1a403a103708c68239b6dcbfc3f82730af
goodcylinder/IMG_8917.JPG 1621d9815274523db222175eb649a78e404174d5
goodcylinder/IMG_8918.JPG 2fa2b5ac9e6437c5291d6994357a8f83576b6ce6
goodcylinder/IMG_8919.JPG 8ba63e34abb3da174f10addf2c222eb854d71526
goodcylinder/IMG_8920.JPG 28ae9fae5db7c60795f132c2dea9587f26d32747
goodcylinder/IMG_8921.JPG a042dbe8c3b031920ed88c9ad7cc5ccbed20b2a0
goodcylinder/IMG_8922.JPG e283146ee33815015ee52e066566ca6b76263779
goodcylinder/IMG_8923.JPG 1a922112c24e33ea2c700e5bc3a32aa62fcbb5f0
goodcylinder/IMG_8924.JPG debe355ad0eabc817fa2fe849eca2228341fbe02
goodcylinder/IMG_8925.JPG f80d1cba4ddaa02364c93987bb51910e677a2c17
goodcylinder/IMG_8926.JPG 995aec2a2829e810029c3c55bff47c44d56a1a0a
goodcylinder/IMG_8927.JPG eb7d32c312e54c3a112705390d7d52bc867502a4
goodcylinder/IMG_8928.JPG 463adb1cecfc104730f5a053b2454c5f53525851
goodcylinder/IMG_8929.JPG 464d6d813279e1cadb69e8993c18c5083af3ae14
goodcylinder/IMG_8930.JPG 3a14cf493ecf8f44b65452347e81a138f1f13e6d
goodcylinder/IMG_8931.JPG 4fece8a9f25ff52ed25b83784def9090a2170f6e
goodcylinder/IMG_8932.JPG b595d6fed61270ca4704fd00ea59efb04aba5a03
goodcylinder/IMG_8933.JPG 27df25af08662c8d06d3986c56fbe4463026acab
goodcylinder/IMG_8934.JPG 6b823703ed20f27e2599c7d87ca53a48a2082fab
goodcylinder/IMG_8935.JPG 4d2abf71f2c54f2f1abba8bd454ef7283d142ab0
goodcylinder/IMG_8936.JPG d03f2ed0ab10e4b39f376c7a78b092033557ae50
goodcylinder/IMG_8937.JPG 21bb45cf2373edc63b7c303c0b1f331993f201aa
goodcylinder/IMG_8938.JPG 7378ddef93dcc47fac5dfe9af177b4125ed46d82
goodcylinder/IMG_8939.JPG e8ec823f22bcbc94e84f97637859f7263891906e
goodcylinder/IMG_8940.JPG 2ecaa34013f8871a2fcdd9dd9a2099f55628622f
goodcylinder/IMG_8941.JPG ee3e7652933f85538af3aa0b76bda449d8d14aeb
goodcylinder/IMG_8942.JPG ef1f6ec44850a481944bafc9831561c40d2269b2
goodcylinder/IMG_8943.JPG 3ee76027d2bf143dca8886312dd151e532f73d85
goodcylinder/IMG_8944.JPG 88eed362320eb00ccbfe1f7e1c7750a77cbad665
goodcylinder/IMG_8945.JPG 1c20c07211f4541fe26408982adb799cb5f51431
goodcylinder/IMG_8946.JPG b391fe00d4c2242c14ba34d87ad1fde4246b3686
goodcylinder/IMG_8947.JPG 56d1ed9fac15a712727eb1f1d8d9018974b89e5b
goodcylinder/IMG_8948.JPG e9e87eb8e21f4457c5079048095fc9eefea36002
goodcylinder/IMG_8949.JPG d0660ea74d9e1eb931a1fa53953bbb5c474229c0
goodcylinder/IMG_8950.JPG 72148f476753d6f8efe9006a0ea502cc569a6cd8
goodcylinder/IMG_8951.JPG 27322cf44ec2bc587288022787c5a4abd0489553
goodcylinder/IMG_8952.JPG ed3c1eb3674f90cd59f8b5077b4f7d4307a18e69
goodcylinder/IMG_8953.JPG 17e2379702398d6a20ead83e87b98b5238114da5
goodcylinder/IMG_8954.JPG fe7dd3dfd89975f5de3e10d4d49b555deabd3663
goodcylinder/IMG_8955.JPG c406e8f9a18b059f98a9ff0d43ef759910718c6c
goodcylinder/IMG_8956.JPG b0ea3804e95f6ba2eb767fbc983a57065c83d75a
goodcylinder/IMG_8957.JPG f6f14b5539bed9bd513a660c847b31fcf9d92717
goodcylinder/IMG_8958.JPG da88d110b4cfd8c86f9224c5f0e2c52b5bfac792
goodcylinder/IMG_8959.JPG c7c0d84097a2a7348403a57c2042404bc88fd3a8
goodcylinder/IMG_8960.JPG 0ebf1a73e3d7adffd5eb08f238cfbcfbbf6fbc5c
goodcylinder/IMG_8961.JPG 4ca92b00052629b0ae7ea58cff71245fd102b241
goodcylinder/IMG_8962.JPG b68e92c84ee20b0f6dc01e1728967ae1b388c755
goodcylinder/IMG_8975.JPG bf82b56da416da8e2c081dcaf1f3080c4cf03ad9
goodcylinder/IMG_8976.JPG 9c28686fa28c2e6407eb60ecfa537a1c34ab5a9f
goodcylinder/IMG_8977.JPG 0253574fe8fa57b97540c1171862214c10110d85
goodcylinder/IMG_8978.JPG b59c9a981f1f13b7363a97932cbbd011bb655761
goodcylinder/IMG_8979.JPG adbf4424b8af99275d1fd70db3669effca170790
goodcylinder/IMG_8980.JPG 521830b1f53a818056d644ea31b590f97bd64c04
goodcylinder/IMG_8981.JPG e878b56507d4660c1f5d6bf98f7ffbce34d7de85
goodcylinder/IMG_8982.JPG 76d3d80d1170486a4e56dd3357d4790dde5fb501
goodcylinder/IMG_8983.JPG 93b8ca0ef22e065f76e2da3e28da8452e43efb32
goodcylinder/IMG_8984.JPG c41c751f92a06b0ada872347a25753bb5d871567
goodcylinder/IMG_8985.JPG 039febbf8fc95fc1d20d4cf76fd9fabbb4aa8266
goodcylinder/IMG_8986.JPG 947f99219988c9755e14b6a7ea137a7ef69b55e5
goodcylinder/IMG_8987.JPG 0143ea847eb323dd0034344bb75671fb811ed003
goodcylinder/IMG_8988.JPG bba9575eb5cb1855ce56de5b458786abb029d8c2
goodcylinder/IMG_8989.JPG 1bc59c9b7df69fbe34a5b4c0082702cb4d30653d
goodcylinder/IMG_8990.JPG e86fafb5d38011061141bb2dcb9736d798dad7a8
goodcylinder/IMG_8991.JPG 851ff44478d2d2789716fefd6168cfbbda6e41f3
goodcylinder/IMG_8992.JPG 64765a2334b38f11f117ee3f9dede49618a3160a
goodcylinder/IMG_8993.JPG 5fd9f6f1c05ccd468de0c54bfffc25c5bf941ca6
goodcylinder/IMG_8994.JPG dfb7635afd09b38fa060db21d83b2eb9fb906b80
goodcylinder/IMG_8995.JPG 4e60f4f96e92b18ecd7aed8d9e14887b35cbbf26
goodcylinder/IMG_8996.JPG 8263f0f329ec119470dd2828b9b8bf18b5218a86
goodcylinder/IMG_8997.JPG a31eb056bc69ba46e0a5abf749cade4a8127a6db
goodcylinder/IMG_8998.JPG 0623c9384a3bf8632a75879d14990d7a3a01973d
goodcylinder/IMG_8999.JPG 99a81f9b2e3577347a2da86f1d5b9721ddd63a40
goodcylinder/IMG_9000.JPG 3e7c721f21dfac418fc606521623cb8a74fe2a34
goodcylinder/IMG_9001.JPG 24f5c8a8982f839e972ca9c998f23f9a0a721f5d
goodcylinder/IMG_9002.JPG 3d5e9f24eee60e33bfe4074175c6d05e538c6b73
goodcylinder/IMG_9003.JPG 885be8b6bcf8be83e32e148b5e6c08d0459d02da
goodcylinder/IMG_9004.JPG 7ccf340ad086dc7a925a5a0334e00b2141b7b5dd
goodcylinder/IMG_9005.JPG 913a6662a1bc4cbdd4ce94457b074bca6388ef2e
goodcylinder/IMG_9006.JPG 9648114ce0d9e49f083bc2b441f690cb178a4c75
goodcylinder/IMG_9007.JPG ea3c5f7e33db39ee42c52f1e2a35cfd766fe075e
goodcylinder/IMG_9008.JPG b6b4f9ad6d8bda609a3309dfe97077bc0193baff
goodcylinder/IMG_9009.JPG c133cb4a8f8ef89b75933c7adea0f24cc6773879
goodcylinder/IMG_9010.JPG 5eef0db2fb8f1cc2e1fb1e758922033851b273f1
goodcylinder/IMG_9011.JPG 3cf4d9b0d8c729df3af0b1f78abb2deea8a40e84
goodcylinder/IMG_9012.JPG 4c3bade4e2e535d3fe1a489be90406791fa4c913
goodcylinder/IMG_9013.JPG 582b1a2dc9b84c1996971ce556952859e8e1a39f
goodcylinder/IMG_9014.JPG 754dd9fc22f9c6ba0efcdb37afacc338002182fc
goodcylinder/IMG_9016.JPG a1b5a89a095edd8bacd0f004ea20ca651f4c0f36
goodcylinder/IMG_9017.JPG 91a21f9c7009087184c42c87835c9b47365b7f30
goodcylinder/IMG_9018.JPG a994ff5cf91e40821b9b950dbcd79edd90182a09
goodcylinder/IMG_9019.JPG cbe6f51f12aaccb5eda55e6ce95147d9bd83b6a8
goodcylinder/IMG_9020.JPG ac404a6c42c7432757a6b3fa123c2362d310b8e7
goodcylinder/IMG_9021.JPG 3f5fcbcecff48e9a1b9624cd3c87c93cec3479a8
goodcylinder/IMG_9022.JPG 3e06bf0fa3603a62cd5814eb2e47d68498646061
goodcylinder/IMG_9023.JPG 2fae1928c6404fb609e40fb3a40c36083c3b5949
goodcylinder/IMG_9024.JPG d051ac62e49eea8fe71995fb9063ea1f56761488
goodcylinder/IMG_9025.JPG 0bd1637fb5142f15cb963df14ccf81222a0b2164
goodcylinder/IMG_9026.JPG ddafec7e55bf7b3dbddd218b138a471fcecb33af
goodcylinder/IMG_9028.JPG e7975d435b1a41e1bef1ea7417f03925076e63d0
goodcylinder/IMG_9029.JPG a97ff5bcbfccb290760ceaa039edf43e90fca505
goodcylinder/IMG_9030.JPG f08a24b173047dd2296da115182e56c0c750c06f
goodcylinder/IMG_9031.JPG ea14ce30804144ecc8fbb3d48416310275d67cea
goodcylinder/IMG_9032.JPG 47785b419257eeefe37efd3f08c8ac1d5e05d87b
goodcylinder/IMG_9033.JPG e97ac4a47fd4ec748de2265cde150025dd19e6ea
goodcylinder/IMG_9034.JPG 13558830fc325d0784cbeb7283317550828c75f5
goodcylinder/IMG_9035.JPG b37f5c79f82e6d6969b63c7a7066957b13f7e453
goodcylinder/IMG_9036.JPG 0f043546f7933f299ee5168918867921cc9091a2
goodcylinder/IMG_9037.JPG 355e3522a3da9cc68f23f02397c8b5e698b9d5ae
goodcylinder/IMG_9038.JPG e3cb9f1910071dca3f0d73c2c2ff806ae93b67a2
goodcylinder/IMG_9039.JPG cddc7f54b32dd26a692664698775527b130cad88
goodcylinder/IMG_9040.JPG 086a9cf13ca991531a93ac28ec0d931036aefc3f
goodcylinder/IMG_9041.JPG dae6590b620f4e38a2610da16279bb45c9b29f47
goodcylinder/IMG_9042.JPG 2168e78e758a6f30fc90539421dd18718534b4e7
goodcylinder/IMG_9043.JPG 54587e5c8bb0107dda81bdf134f331eb7b4cbc6c
goodcylinder/IMG_9044.JPG 70276aad9da0822b060938ba211589d800743b6c
goodcylinder/IMG_9045.JPG aecacd075e44c73f30f22eab3b98defbb5a86c37
goodcylinder/IMG_9046.JPG 5f4d5891593ebc1f1fc628817145d0633106cd3d
goodcylinder/IMG_9047.JPG aebce1b311a3143f7cbfe6fc4b33e0889785d935
goodcylinder/IMG_9048.JPG 606d75e5e86205ed07b2c443f9d232f72aec9025
goodcylinder/IMG_9049.JPG aca0aeb129fa068129077e0afa31d8b9cfe66469
goodcylinder/IMG_9050.JPG ecab51eb662d28e51a2ef391989c005ca2692f8c
goodcylinder/IMG_9051.JPG 0ffe718f6bdd68b4566773b0d7fb6aebbe859654
goodcylinder/IMG_9052.JPG 01b67496aa219333eb8a97f2e32f93181a04e8de
goodcylinder/IMG_9053.JPG aec901563106494e47f679bc62720cdf15c4cf83
goodcylinder/IMG_9054.JPG fa27e36abc164ca21c9730a812cab66bcaefefd4
goodcylinder/IMG_9055.JPG 817f62bb4379eed62018af7ed42c8accd86ada22
goodcylinder/IMG_9056.JPG 0db593e5dc002e3d0b2bc9687689bf9659781910
goodcylinder/IMG_9057.JPG e9b3fcc5be6be0b38460f574324c145ef0c9eebd
goodcylinder/IMG_9058.JPG 2330323b16ed57f98b1f279bf45a1311414da1e7
goodcylinder/IMG_9059.JPG 4a3265cfe68d9c2dc1ba8e0cd8493cd7908fdd4a
goodcylinder/IMG_9060.JPG 22b02a2938b0e8c19cd038999fdf4c7288069174
goodcylinder/IMG_9061.JPG dd394b32e258cbb1342d1fe13bee4a021f5a70a7
goodcylinder/IMG_9062.JPG 54a176ed410e5b065d9881ea958c7f77e67f240b
goodcylinder/IMG_9063.JPG e486cfd5af92ade0ea00e65aac60c095bb7ec6cf
"""


def fetch(url):
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "AMdataset-reproduction"})
            with urlopen(request, timeout=90) as response:
                return response.read()
        except (OSError, HTTPException):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def blob_hash(data):
    return sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def download(entry, root):
    relative = Path(entry["path"])
    target = root / relative
    if target.is_file() and blob_hash(target.read_bytes()) == entry["sha"]:
        return
    url = f"https://raw.githubusercontent.com/{REPO}/{REVISION}/{quote(relative.as_posix())}"
    data = fetch(url)
    if blob_hash(data) != entry["sha"]:
        raise ValueError(f"Git blob checksum mismatch: {relative}")
    pending = target.with_suffix(".part")
    pending.write_bytes(data)
    pending.replace(target)


def prepare(root):
    entries = [{"path": f"dataset/{path}", "sha": digest}
               for path, digest in (line.split() for line in DATASET_BLOBS.splitlines())]
    for category in CLASSES:
        (root / "dataset" / category).mkdir(parents=True, exist_ok=True)
    print(f"Verifying or downloading {len(entries)} pinned images.", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, _ in enumerate(pool.map(lambda entry: download(entry, root), entries), 1):
            if index % 25 == 0 or index == len(entries):
                print(f"Verified {index}/{len(entries)}", flush=True)


def read_image(path):
    # imdecode also handles Windows paths containing non-ASCII characters.
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def load_split(root, seed):
    paths, labels = [], []
    for label, name in enumerate(CLASSES):
        files = sorted(p for p in (root / name).glob("*")
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise ValueError(f"Expected labeled images under {root / name}")
        paths.extend(files)
        labels.extend([label] * len(files))
    labels = np.asarray(labels)
    train, remaining = train_test_split(np.arange(len(labels)), test_size=0.30,
                                        random_state=seed)
    val, test = train_test_split(remaining, test_size=0.66, random_state=seed)
    split = {"train": train, "val": val, "test": test}
    for name, ids in split.items():
        if len(np.unique(labels[ids])) != 2:
            raise ValueError(f"{name} must contain examples of both classes.")
    return paths, labels, split


def build_extractor(weights):
    base = keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3),
                                         include_top=False, weights=weights)
    base.trainable = False
    rgb = keras.Input((SIZE, SIZE, 3), dtype="uint8", name="rgb")
    x = keras.layers.Rescaling(1.0 / 255)(rgb)
    x = base(x, training=False)
    x = keras.layers.AveragePooling2D(pool_size=4)(x)
    features = keras.layers.Flatten()(x)
    return keras.Model(rgb, features, name="frozen_features")


def build_head(feature_dim, optimizer, learning_rate):
    head = keras.Sequential([
        keras.Input((feature_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(2, activation="softmax"),
    ], name="classifier")
    head.compile(optimizer=OPTIMIZERS[optimizer](learning_rate=learning_rate),
                 loss="binary_crossentropy", metrics=["accuracy"])
    return head


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def score(labels, probabilities):
    actual = labels == DEFECT
    predicted = probabilities.argmax(axis=1) == DEFECT
    confidence = probabilities[:, DEFECT]
    tn, fp, fn, tp = metrics.confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    return {
        "n": len(actual), "accuracy": metrics.accuracy_score(actual, predicted),
        "precision_defect": metrics.precision_score(actual, predicted, zero_division=0),
        "recall_defect": metrics.recall_score(actual, predicted, zero_division=0),
        "f1_defect": metrics.f1_score(actual, predicted, zero_division=0),
        "specificity": float(tn / (tn + fp)), "auc_roc": metrics.roc_auc_score(actual, confidence),
        "ap_defect": metrics.average_precision_score(actual, confidence),
        "mcc": metrics.matthews_corrcoef(actual, predicted),
        "kappa": metrics.cohen_kappa_score(actual, predicted),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def transform(image, condition):
    if condition == "clean":
        return image
    if condition in ("brightness0.5", "brightness1.5"):
        factor = 0.5 if condition == "brightness0.5" else 1.5
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if condition == "blur9_sigma2.0":
        return cv2.GaussianBlur(image, (9, 9), 2.0)
    if condition == "jpeg_quality15":
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 15])
        if not ok:
            raise ValueError("OpenCV could not encode the JPEG test view.")
        return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if condition == "resize56":
        return cv2.resize(cv2.resize(image, (56, 56)), (SIZE, SIZE))
    if condition == "rotate10_reflect":
        matrix = cv2.getRotationMatrix2D(((SIZE - 1) / 2, (SIZE - 1) / 2), 10, 1)
        return cv2.warpAffine(image, matrix, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT_101)
    raise ValueError(f"Unknown condition: {condition}")


def features(extractor, data, rows, conditions):
    result = {c: [] for c in conditions}
    for start in range(0, len(rows), 24):
        images = [read_image(data / path) for path, _ in rows[start:start + 24]]
        for condition in conditions:
            batch = np.stack([transform(im, condition) for im in images])
            result[condition].append(extractor.predict_on_batch(batch))
    return {c: np.concatenate(batches) for c, batches in result.items()}


def fit_head(x, y, orders, views, seed, arm, optimizer):
    keras.utils.set_random_seed(seed)
    head = build_head(x.shape[-1], optimizer, LR)
    initial = head.get_weights()
    history = []
    for epoch, order in enumerate(orders):
        head.reset_metrics()
        for start in range(0, len(order), BATCH):
            ids = order[start:start + BATCH]
            chosen = views[epoch, start:start + BATCH] if arm == "augmented" else 0
            logs = head.train_on_batch(x[chosen, ids], y[ids], return_dict=True)
        history.append({"arm": arm, "seed": seed, "epoch": epoch + 1,
                        **{key: float(value) for key, value in logs.items()}})
    updates = int(keras.ops.convert_to_numpy(head.optimizer.iterations))
    if updates != EPOCHS * ceil(len(y) / BATCH):
        raise RuntimeError(f"Unexpected optimizer updates for {arm}, seed {seed}: {updates}")
    return head, initial, history, updates


def evaluate(heads, encoded, rows, part):
    labels = np.array([label for _, label in rows])
    summaries, predictions = [], []
    for (arm, seed), head in heads.items():
        for condition, vectors in encoded.items():
            p = head.predict_on_batch(vectors)
            tags = {"split": part, "arm": arm, "seed": seed, "condition": condition}
            summaries.append({**tags, **score(labels, p)})
            predictions.extend(
                {**tags, "path": path.as_posix(), "actual": CLASSES[label],
                 "predicted": CLASSES[int(prob.argmax())], "p_defect": float(prob[0])}
                for (path, label), prob in zip(rows, p, strict=True))
    return summaries, predictions


def select(rows):
    by_arm = {}
    for arm in ARMS:
        subset = [r for r in rows if r["arm"] == arm]
        by_arm[arm] = {
            "mean_f1": mean(r["f1_defect"] for r in subset),
            "clean_f1": mean(r["f1_defect"] for r in subset if r["condition"] == "clean"),
            "worst_recall": min(mean(r["recall_defect"] for r in subset if r["condition"] == c)
                                for c in CONDITIONS),
        }
    clean, aug = (by_arm[arm] for arm in ARMS)
    adopt = (aug["mean_f1"] >= clean["mean_f1"] + 0.01
             and aug["clean_f1"] >= clean["clean_f1"]
             and aug["worst_recall"] >= clean["worst_recall"])
    return {"selected_arm": "augmented" if adopt else "clean", "rule": RULE, "validation": by_arm}


def summarize(metrics):
    for part in ("val", "test"):
        for arm in ARMS:
            subset = [r for r in metrics if r["split"] == part and r["arm"] == arm]
            for condition in (*CONDITIONS, "seven_condition_mean"):
                # Average conditions within each seed before estimating the across-seed SD.
                by_seed = [[r for r in subset if r["seed"] == seed
                            and (r["condition"] == condition or condition == "seven_condition_mean")]
                           for seed in SEEDS]
                accuracies = [mean(r["accuracy"] for r in rows) for rows in by_seed]
                yield {"split": part, "arm": arm, "condition": condition,
                       "accuracy_mean": mean(accuracies), "accuracy_sd": stdev(accuracies),
                       "recall_mean": mean(r["recall_defect"] for rows in by_seed for r in rows),
                       "f1_mean": mean(r["f1_defect"] for rows in by_seed for r in rows)}


def train(args):
    if keras.backend.backend() == "torch":
        import torch
        torch.set_num_threads(4)
    cv2.setNumThreads(1)
    paths, labels, split = load_split(args.data, SEEDS[0])
    groups = {part: [(paths[i].relative_to(args.data), int(labels[i])) for i in ids]
              for part, ids in split.items()}
    if tuple(map(len, groups.values())) != (189, 27, 55):
        raise ValueError("Expected the pinned Study One dataset (189 train, 27 validation, 55 test).")
    args.out.mkdir(parents=True, exist_ok=False)
    save_csv(args.out / "split.csv", [
        {"split": part, "path": p.as_posix(), "label": CLASSES[label]}
        for part, rows in groups.items() for p, label in rows])
    save_json(args.out / "protocol.json", {
        "dataset_revision": REVISION, "keras": keras.__version__, "backend": keras.backend.backend(),
        "seeds": SEEDS, "epochs": EPOCHS, "learning_rate": LR, "batch_size": BATCH,
        "optimizers": args.optimizer, "training_views": VIEWS, "test_conditions": CONDITIONS,
        "reference_accuracy": PAPER, "selection_rule": RULE, "output_classes": CLASSES,
        "weights_sha256": (hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()
                           if Path(args.weights).is_file() else None),
    })
    keras.utils.set_random_seed(SEEDS[0])
    extractor = build_extractor(args.weights)
    print("Extracting training and validation features.", flush=True)
    train_views = features(extractor, args.data, groups["train"], VIEWS)
    x = np.stack([train_views[v] for v in VIEWS])
    y = keras.utils.to_categorical([label for _, label in groups["train"]], 2)
    val_x = features(extractor, args.data, groups["val"], CONDITIONS)
    heads_by_optimizer, validation, fits, histories = {}, {}, [], []
    for optimizer in args.optimizer:
        heads = {}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            orders = np.stack([rng.permutation(len(y)) for _ in range(EPOCHS)])
            views = np.random.default_rng(seed + 10000).integers(
                0, len(VIEWS), size=orders.shape, dtype=np.int32)
            for arm in ARMS:
                head, initial, history, updates = fit_head(x, y, orders, views, seed, arm, optimizer)
                if arm == "clean":
                    initial_reference = initial
                else:
                    for earlier, current in zip(initial_reference, initial, strict=True):
                        np.testing.assert_array_equal(earlier, current)
                heads[arm, seed] = head
                histories.extend({"optimizer": optimizer, **r} for r in history)
                fits.append({
                    "optimizer": optimizer, "arm": arm, "seed": seed, "updates": updates,
                    "initial_sha256": hashlib.sha256(b"".join(a.tobytes() for a in initial)).hexdigest(),
                    "image_order_sha256": hashlib.sha256(orders.tobytes()).hexdigest(),
                })
                print(f"{optimizer} / {arm} / seed {seed}: {updates} updates", flush=True)
        metrics_val, predictions_val = evaluate(heads, val_x, groups["val"], "val")
        validation[optimizer] = (metrics_val, predictions_val)
        heads_by_optimizer[optimizer] = heads

    # Record each outcome before test feature extraction. The report compares
    # the predefined augmented arm for every optimizer, irrespective of adoption.
    save_json(args.out / "validation_decisions.json", {
        optimizer: select(metrics_val) for optimizer, (metrics_val, _) in validation.items()})
    save_csv(args.out / "history.csv", histories)
    save_csv(args.out / "fits.csv", fits)
    print("Extracting test features.", flush=True)
    test_x = features(extractor, args.data, groups["test"], CONDITIONS)
    sample = np.stack([read_image(args.data / path) for path, _ in groups["test"][:4]])
    all_metrics, all_predictions, all_summaries, paper_comparison, model_checks = [], [], [], [], []
    for optimizer, heads in heads_by_optimizer.items():
        test_metrics, test_predictions = evaluate(heads, test_x, groups["test"], "test")
        val_metrics, val_predictions = validation[optimizer]
        summaries = list(summarize(val_metrics + test_metrics))
        all_metrics.extend({"optimizer": optimizer, **r} for r in val_metrics + test_metrics)
        all_predictions.extend({"optimizer": optimizer, **r} for r in val_predictions + test_predictions)
        all_summaries.extend({"optimizer": optimizer, **r} for r in summaries)
        row = next(r for r in summaries if (r["split"], r["arm"], r["condition"])
                   == ("test", "augmented", "clean"))
        paper_comparison.append({
            "optimizer": optimizer, "augmented_mean_accuracy": row["accuracy_mean"],
            "augmented_accuracy_sd": row["accuracy_sd"], "paper_accuracy": PAPER[optimizer],
            "difference_pp": 100 * (row["accuracy_mean"] - PAPER[optimizer]),
        })
        print(f"{optimizer}: augmented clean accuracy {row['accuracy_mean']:.4%} "
              f"vs paper {PAPER[optimizer]:.4%}", flush=True)
    save_csv(args.out / "paper_comparison.csv", paper_comparison)
    save_csv(args.out / "metrics.csv", all_metrics)
    save_csv(args.out / "predictions.csv", all_predictions)
    save_csv(args.out / "summary.csv", all_summaries)

    for optimizer, heads in heads_by_optimizer.items():
        head = heads["augmented", SEEDS[0]]
        model = keras.Model(extractor.input, head(extractor.output), name="mobilenetv2_defects")
        destination = args.out / f"augmented_{optimizer}_seed42.keras"
        model.save(destination)
        direct = model.predict_on_batch(sample)
        cached = head.predict_on_batch(test_x["clean"][:4])
        recovered = keras.models.load_model(destination, compile=False).predict_on_batch(sample)
        np.testing.assert_allclose(direct, cached, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(direct, recovered, atol=1e-6, rtol=1e-5)
        model_checks.append({"optimizer": optimizer, "file": destination.name,
                             "parameters": model.count_params(), "checks": len(sample),
                             "max_reload_error": float(np.max(np.abs(direct - recovered)))})
    save_json(args.out / "model_checks.json", model_checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    dataset = commands.add_parser("prepare", help="Download the pinned AMdataset images")
    dataset.add_argument("--root", type=Path, default=Path("AMdataset"))
    run = commands.add_parser("train", help="Run paired training and compare augmentation with Study One")
    run.add_argument("--data", type=Path, default=Path("AMdataset/dataset"))
    run.add_argument("--out", type=Path, default=Path("runs"))
    run.add_argument("--weights", default="imagenet", help="imagenet or local Keras no-top .h5")
    run.add_argument("--optimizer", nargs="+", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    infer = commands.add_parser("predict", help="Classify an image with a trained .keras model")
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root)
    elif args.command == "train":
        if len(set(args.optimizer)) != len(args.optimizer):
            parser.error("--optimizer must not repeat an optimizer")
        train(args)
    else:
        model = keras.models.load_model(args.model, compile=False)
        p = model.predict_on_batch(read_image(args.image)[None])[0]
        print(json.dumps({"class": CLASSES[int(p.argmax())], "p_defect": float(p[DEFECT])}))


if __name__ == "__main__":
    main()
