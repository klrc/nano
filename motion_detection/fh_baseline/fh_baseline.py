from numpy import uint32


def detect_motion(some_video):
    return None


def md_detector(img1, img2, threshold, threshold_pix_num):
    '''
        img1: yuv luma gray image
        img2: next yuv luma gray image
        threshold:
        threshold_pix_num:
    '''
    i, j, k, kw, cnt = uint32(0)
    h, w = img1.shape

    for j in range(h):
        for i in range(w):

    # 			tempX = disW;
    # 			tempY = disH;
    # 			offsetX = reminderW;
    # 			offsetY = reminderH;
    # 			if (i < reminderW)
    # 			{
    # 				tempX += 1;
    # 				offsetX = i;
    # 			}
    # 			if (j < reminderH)
    # 			{
    # 				tempY += 1;
    # 				offsetY = j;
    # 			}
    # 			gCalSize = tempX * tempY;//g_mdCalcParam.size
    # 			gUValue = 0;//g_mdCalcParam.uValue
    # 			numGreat = 0;
    # 			gUValueLess = 0;
    # 			numLess = 0;
    # 			for (k = 0; k < tempY; k++)
    # 			{
    # 				tempOffset = j * disH + offsetY + k;
    # 				//memcpy(pValue + k * tempX, pInYmean + i * disW + offsetX + tempOffset * temp_w, tempX);//g_mdCalcParam.

    # 				//memcpy(pValue2 + k * tempX, pInYmean2 + i * disW + offsetX + tempOffset * temp_w, tempX);

    # 				pValue1 = pInYmean + i * disW + offsetX + tempOffset * pitchIn;// temp_w;
    # 				pValue2 = pInYmean2 + i * disW + offsetX + tempOffset * pitchIn;//temp_w;
    # 				for (kw = 0; kw < tempX; ++kw)
    # 				{
    # 					diffValue = abs(pValue1[kw] - pValue2[kw]);

    # 					if (diffValue > threashold)
    # 					{
    # 						gUValue += diffValue;//g_mdCalcParam.uValue, g_mdCalcParam.
    # 						numGreat++;
    # 					}
    # 					else
    # 					{
    # 						gUValueLess += diffValue;
    # 						numLess++;
    # 					}
    # 				}
    # 			}

    # 			//for (k = 0; k < gCalSize; k++)//g_mdCalcParam.size
    # 			//{
    # 			//	diffValue = abs(pValue[k] - pValue2[k]);
    # 			//	if (diffValue > threashold)
    # 			//	{
    # 			//		gUValue += diffValue;//g_mdCalcParam.uValue, g_mdCalcParam.
    # 			//		numGreat++;
    # 			//	}
    # 			//}
    # 			cnt = i + j * pitchOut;// horNum; //my: i*verNum + j;//
    # 			if (numGreat > 0)// gUValue) //g_mdCalcParam.size)
    # 			{
    # 				if (numGreat > thresholdPixNum) {
    # 					gUValue = floor((float)gUValue / numGreat + 0.5);
    # 					gUValue = gUValue / numGreat;// gCalSize;//g_mdCalcParam.uValue = g_mdCalcParam.uValue / g_mdCalcParam.size;
    # 					*(pOutYmean + cnt) = (FY_U8)gUValue;
    # 				}
    # 				else {
    # 					gUValueLess = floor((float)(gUValue+gUValueLess) / (numLess+ numGreat) + 0.5);
    # 					*(pOutYmean + cnt) = (FY_U8)gUValueLess;
    # 				}
    # 			}
    # 			else
    # 			{
    # 				gUValueLess = floor((float)gUValueLess / numLess + 0.5);
    # 				gUValueLess = gUValueLess / numLess;
    # 				*(pOutYmean + cnt) = (FY_U8)gUValueLess;
    # 			}
    # 			//*(pOutYmean + cnt) = (FY_U8)g_mdCalcParam.uValue;
    return None


if __name__ == "__main__":
    print("helloworld")
