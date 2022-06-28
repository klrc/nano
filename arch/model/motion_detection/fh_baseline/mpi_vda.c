#include "mpi_vda.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "ioe_drv_ioc.h"
#include "mpi_sys.h"
#include "mpi_vb.h"
#include "mpi_vpss.h"

#define CURRENT_VERSION "V0.2.8"

#define ERR_NOT_INIT -1
#define MAX_CNT 256

#define GET_BITS_ALIGNMENT(p, bits) (p + (bits - 1)) & (~((bits - 1)))
#define GET_DIS_OFFSET(x, y) (x) / (y)
#define GET_REMINDER_OFFSET(x, y) (x) % (y)
#define GET_MACRO_BLOCK_NUM(x, bits) x / bits
#define MAX_VALUE(a, b) ((a) > (b) ? (a) : (b))
#define MIN_VALUE(a, b) ((a) < (b) ? (a) : (b))

#define DOWN_ALLIGN(p, bits) (p) & (~((bits - 1)))
#define UP_ALIGNTO(addr, edge) ((addr + edge - 1) & ~(edge - 1))
#define DOWN_ALIGNTO(x, align) (((unsigned int)(x)) & (~((align)-1)))
#define BIT_ALLIGN_BYTES(a, n, b) a % 8 ? (((a >> n) << n) + b) : a

#define CHECK_BUFFER_SIZE 2
#define FILE_NAME_LEN 30
#define VPU_YMEAN_ALIGN 8
#define COMP_VALUE 0

//#define DEBUG_SAVE_FILE			FY_TRUE
//#define DEBUG_READ_YC_FILE		FY_TRUE

#ifdef DEBUG_READ_YC_FILE
#define TEST_YC_FILE_NAME1 "/dbghik/vpss_grp0_chn0_240x272_index1_cnt1.1.yuv"
#define TEST_YC_FILE_NAME2 "/dbghik/vpss_grp0_chn0_240x272_index1_cnt1.yuv"
static FY_BOOL g_readFile_flag = FY_TRUE;
#endif

typedef enum { MD_MMZ_YC, MD_MMZ_DYC, MD_MMZ_OUTPUT } MD_MEM_TYPE;

typedef struct _VDA_Mem_s {
    FY_CHAR name[FILE_NAME_LEN];
    FY_VOID* pVirAddr;
    FY_U32 u32DataLen;
    FY_U32 u32PhyAddr;
    FY_U32 u32Width;
    FY_U32 u32Height;
    FY_U32 u32DsMode;
} VDA_MEM_S;

typedef struct _FY_VDA_MD_CFG {
    pthread_mutex_t mdMutex;
    FY_BOOL bInit;
    FY_U32 u32OutSize;
    FY_U32 u32Ord;
    FY_U32 uFrmDelay;
    FY_U32 uHorMacNum;
    FY_U32 uVerMacNum;
    FY_U64 u64LstPts;
    FY_U64 u64CurPts;

    FY_U8* pMdStatus;

    VDA_MEM_S Yc[CHECK_BUFFER_SIZE];
    VDA_MEM_S Dyc[CHECK_BUFFER_SIZE];
    VDA_MEM_S Dvalue;
    FY_VDA_MD_CFG_t mdCfg;
    FY_VDA_INIT_cfg initParam;
    FY_MDConfig_Ex_Result_t cachePool;
} FY_VDA_MD_CFG;

typedef struct _FY_VDA_CD_CFG {
    pthread_mutex_t cdMutex;
    FY_BOOL bInit;
    FY_U32 uFrmDelay;
    FY_U32 u32Width;
    FY_U32 u32Height;
    FY_U32 count[MAX_CNT];
    FY_U32 u32FailCnt;
    FY_U32 cdResult;
    FY_U64 u64LstPts;
    FY_U64 u64CurPts;

    VDA_MEM_S Yc;
    FY_VDA_INIT_cfg initParam;
    FY_VDA_CD_INFO_t cdCfg;
} FY_VDA_CD_CFG;

typedef struct {
    FY_U32 uValue;
    FY_U32 size;
    FY_U8* pValue;
} VDA_MD_Macro_s;

typedef struct _vda_file_info {
    FY_CHAR name[FILE_NAME_LEN];
    FY_CHAR* pAddr;
    FY_U32 len;
} VDA_FILE_INFO;

typedef struct _vda_thread_s {
    FY_BOOL bStart;
    FY_U32 u32Type;
    FY_U32 param1;
    pthread_t t_id;
} VDA_THREAD_S;

typedef struct _vda_hw_s {
    FY_BOOL bOpen;
    FY_S32 sFd;
} VDA_HW_S;

typedef struct _DealYc_param {
    SIZE_S inRect;
    SIZE_S outRect;
    FY_VOID* pYcApr;
    FY_U16 u16OutMode;
    FY_U8* pOutput;
    FY_U32 u32thed;
} DEAL_YC_PARAM;

FY_VDA_MD_CFG g_MdCfgInfo[MAX_MD_CHANNEL_NUM];
FY_VDA_CD_CFG g_CoverCfgInfo[MAX_MD_CHANNEL_NUM];

static VDA_THREAD_S g_VdaThrInfo[VDA_TYPE_CD + 1] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
static VDA_HW_S g_VdaHwInfo = {0, 0};

#define GetDiffValue_Byte(x, y) abs((FY_U8)x - (FY_U8)y)

#define CHECK_ID_IS_VALID(id)                   \
    do {                                        \
        if (id >= MAX_MD_CHANNEL_NUM) {         \
            printf("id is invalid!(%d)\n", id); \
            return FY_FAILURE;                  \
        }                                       \
    } while (0)

#define VDA_CHECK_PTR_NULL(ptr)                                        \
    do {                                                               \
        if (!ptr) {                                                    \
            printf("[%s]line %d param is NULL\n", __func__, __LINE__); \
            return FY_FAILURE;                                         \
        }                                                              \
    } while (0)

#define VDA_MD_IS_INIT(id)                                                    \
    do {                                                                      \
        if (!g_MdCfgInfo[id].bInit) {                                         \
            printf("[%s]line %d (id-%d) not init\n", __func__, __LINE__, id); \
            return FY_FAILURE;                                                \
        }                                                                     \
    } while (0)

#define VDA_CD_IS_INIT(id)                                                    \
    do {                                                                      \
        if (!g_CoverCfgInfo[id].bInit) {                                      \
            printf("[%s]line %d (id-%d) not init\n", __func__, __LINE__, id); \
            return FY_FAILURE;                                                \
        }                                                                     \
    } while (0)

#define VDA_Alloc_Memory(pAtr, size)                                          \
    do {                                                                      \
        if (pAtr == FY_NULL) {                                                \
            pAtr = malloc(size);                                              \
            if (pAtr == FY_NULL) {                                            \
                printf("[%s]%d: alloc memory failed!\n", __func__, __LINE__); \
                goto exit;                                                    \
            }                                                                 \
        }                                                                     \
        memset(pAtr, 0, size);                                                \
    } while (0)

#define VDA_Free_Memory(pAtr)  \
    do {                       \
        if (pAtr != FY_NULL) { \
            free(pAtr);        \
            pAtr = FY_NULL;    \
        }                      \
    } while (0)

#define VDA_OPEN_FILE(file, filename, flag)                       \
    do {                                                          \
        file = fopen(filename, flag);                             \
        if (file == NULL) {                                       \
            printf("[%s]open %s fail\n", __FUNCTION__, filename); \
            goto exit;                                            \
        }                                                         \
    } while (0)

#define VDA_WRITE_FILE(file, ptr, len)              \
    do {                                            \
        if (file && ptr) fwrite(ptr, 1, len, file); \
    } while (0)

#define VDA_READ_FILE(file, ptr, len, readlen)                                               \
    do {                                                                                     \
        if (file && ptr) readlen = fread(ptr, 1, len, file);                                 \
        if (readlen != len) printf("[vda][error]read file error len=%d-%d\n", len, readlen); \
    } while (0)

#define VDA_CLOSE_FILE(file)            \
    do {                                \
        if (file != NULL) fclose(file); \
        file = FY_NULL;                 \
    } while (0)

#define VDA_LOG(bOpen, msg, ...)    \
    if (bOpen) {                    \
        printf(msg, ##__VA_ARGS__); \
    }

static FY_CHAR* show_MD_VERSION(FY_U32 id) {
    static char vers[70];

    if (id == 0) {
        sprintf(vers, "MD version: \t\t%s(time:%s   %s)\n", CURRENT_VERSION, __DATE__, __TIME__);
        fprintf((FILE*)stderr, vers);
    }
    return vers;
}

FY_S32 VDA_Free_Memz(FY_U32 id, MD_MEM_TYPE mType, FY_U32 u32Ord) {
    FY_S32 ret = FY_SUCCESS;
    FY_U32 u32PhyAddr = 0;
    FY_VOID* pVirAddr = FY_NULL;

    switch (mType) {
        case MD_MMZ_YC: {
            u32PhyAddr = g_MdCfgInfo[id].Yc[u32Ord].u32PhyAddr;
            pVirAddr = g_MdCfgInfo[id].Yc[u32Ord].pVirAddr;
            if (u32PhyAddr && pVirAddr) {
                ret = FY_MPI_SYS_MmzFree(u32PhyAddr, pVirAddr);
            }
            g_MdCfgInfo[id].Yc[u32Ord].u32PhyAddr = 0;
            g_MdCfgInfo[id].Yc[u32Ord].pVirAddr = FY_NULL;
            g_MdCfgInfo[id].Yc[u32Ord].u32DataLen = 0;
        } break;
        case MD_MMZ_DYC: {
            u32PhyAddr = g_MdCfgInfo[id].Dyc[u32Ord].u32PhyAddr;
            pVirAddr = g_MdCfgInfo[id].Dyc[u32Ord].pVirAddr;
            if (u32PhyAddr && pVirAddr) {
                ret = FY_MPI_SYS_MmzFree(u32PhyAddr, pVirAddr);
            }
            g_MdCfgInfo[id].Dyc[u32Ord].u32PhyAddr = 0;
            g_MdCfgInfo[id].Dyc[u32Ord].pVirAddr = FY_NULL;
            g_MdCfgInfo[id].Dyc[u32Ord].u32DataLen = 0;
        } break;
        case MD_MMZ_OUTPUT: {
            u32PhyAddr = g_MdCfgInfo[id].Dvalue.u32PhyAddr;
            pVirAddr = g_MdCfgInfo[id].Dvalue.pVirAddr;
            if (u32PhyAddr && pVirAddr) {
                ret = FY_MPI_SYS_MmzFree(u32PhyAddr, pVirAddr);
            }
            g_MdCfgInfo[id].Dvalue.u32PhyAddr = 0;
            g_MdCfgInfo[id].Dvalue.pVirAddr = FY_NULL;
            g_MdCfgInfo[id].Dvalue.u32DataLen = 0;
        } break;

        default:
            ret = FY_FAILURE;
            break;
    }

    return ret;
}

FY_S32 VDA_Malloc_Memz(FY_U32 id, MD_MEM_TYPE mType, FY_U32 size) {
    FY_S32 ret = FY_SUCCESS;
    FY_U32 u32Ord = 0;
    FY_U32 u32PhyAddr = 0, u32DataLen;
    FY_VOID* pVirAddr = FY_NULL;
    FY_CHAR* pName = FY_NULL;
    FY_BOOL bLog = FY_FALSE;

    bLog = g_MdCfgInfo[id].initParam.bOpenLog;
    switch (mType) {
        case MD_MMZ_YC: {
            u32Ord = g_MdCfgInfo[id].u32Ord;
            pName = &g_MdCfgInfo[id].Yc[u32Ord].name[0];

            u32DataLen = g_MdCfgInfo[id].Yc[u32Ord].u32DataLen;
            VDA_LOG(bLog, "[%s]line[%d][id%d]ord[%d]size[%d]:[%d]\n", __func__, __LINE__, id, u32Ord, u32DataLen, size);
            if (u32DataLen != size) {
                VDA_Free_Memz(id, mType, u32Ord);
            }
            u32PhyAddr = g_MdCfgInfo[id].Yc[u32Ord].u32PhyAddr;
            pVirAddr = g_MdCfgInfo[id].Yc[u32Ord].pVirAddr;
            if (!u32PhyAddr && !pVirAddr) {
                ret = FY_MPI_SYS_MmzAlloc(&u32PhyAddr, &pVirAddr, pName, FY_NULL, size);
            }
            if (ret == FY_SUCCESS) {
                memset(pVirAddr, 0, size);
                g_MdCfgInfo[id].Yc[u32Ord].u32PhyAddr = u32PhyAddr;
                g_MdCfgInfo[id].Yc[u32Ord].pVirAddr = pVirAddr;
                g_MdCfgInfo[id].Yc[u32Ord].u32DataLen = size;
            }
        } break;
        case MD_MMZ_DYC: {
            u32Ord = g_MdCfgInfo[id].u32Ord;
            pName = &g_MdCfgInfo[id].Dyc[u32Ord].name[0];

            u32DataLen = g_MdCfgInfo[id].Dyc[u32Ord].u32DataLen;
            VDA_LOG(bLog, "[%s]line[%d][id%d]ord[%d]size[%d]:[%d]\n", __func__, __LINE__, id, u32Ord, u32DataLen, size);
            if (u32DataLen != size) {
                VDA_Free_Memz(id, mType, u32Ord);
            }
            u32PhyAddr = g_MdCfgInfo[id].Dyc[u32Ord].u32PhyAddr;
            pVirAddr = g_MdCfgInfo[id].Dyc[u32Ord].pVirAddr;
            if (!u32PhyAddr && !pVirAddr) {
                ret = FY_MPI_SYS_MmzAlloc(&u32PhyAddr, &pVirAddr, pName, FY_NULL, size);
            }
            if (ret == FY_SUCCESS) {
                memset(pVirAddr, 0, size);
                g_MdCfgInfo[id].Dyc[u32Ord].u32PhyAddr = u32PhyAddr;
                g_MdCfgInfo[id].Dyc[u32Ord].pVirAddr = pVirAddr;
                g_MdCfgInfo[id].Dyc[u32Ord].u32DataLen = size;
            }
        } break;
        case MD_MMZ_OUTPUT: {
            pName = &g_MdCfgInfo[id].Dvalue.name[0];

            u32DataLen = g_MdCfgInfo[id].Dvalue.u32DataLen;
            VDA_LOG(bLog, "[%s]line[%d][id%d]ord[%d]size[%d]:[%d]\n", __func__, __LINE__, id, u32Ord, u32DataLen, size);
            if (u32DataLen != size) {
                VDA_Free_Memz(id, mType, u32Ord);
            }
            u32PhyAddr = g_MdCfgInfo[id].Dvalue.u32PhyAddr;
            pVirAddr = g_MdCfgInfo[id].Dvalue.pVirAddr;
            if (!u32PhyAddr && !pVirAddr) {
                ret = FY_MPI_SYS_MmzAlloc(&u32PhyAddr, &pVirAddr, pName, FY_NULL, size);
            }
            if (ret == FY_SUCCESS) {
                memset(pVirAddr, 0, size);
                g_MdCfgInfo[id].Dvalue.u32PhyAddr = u32PhyAddr;
                g_MdCfgInfo[id].Dvalue.pVirAddr = pVirAddr;
                g_MdCfgInfo[id].Dvalue.u32DataLen = size;
            }
        } break;

        default:
            ret = FY_FAILURE;
            break;
    }
    VDA_LOG(bLog, "[%s]line[%d][id%d]ord[%d]addr[%x]:[%p],size[%d]\n", __func__, __LINE__, id, u32Ord, u32PhyAddr,
            pVirAddr, size);
    return ret;
}

FY_VOID VDA_CHECK_SIZE_IS_ENQ(FY_U32 NewSize, FY_U32* OldSize, FY_VOID** ptr) {
    FY_VOID* pAtr = FY_NULL;

    if (NewSize != *OldSize) {
        *OldSize = NewSize;
        VDA_Free_Memory(*ptr);
        VDA_Alloc_Memory(pAtr, NewSize);
        *ptr = pAtr;
    }
exit:
    return;
}

static FY_S32 _VDA_MdPrtPoint(FY_MDConfig_Ex_Result_t* pstVdaData) {
    FY_S32 i, j, k;
    FY_U8* pu8Addr;
    FILE* fp = stdout;
    FY_U32 u32X, u32Y, u32W, u32H, u32Ver, u32Hor;

    //	fprintf(fp, "===== %s =====\n", __FUNCTION__);

    for (i = 0; i < pstVdaData->cached_number; i++) {
        u32X = pstVdaData->rect[i].fTopLeftX;
        u32Y = pstVdaData->rect[i].fTopLeftY;
        u32W = pstVdaData->rect[i].fWidth;
        u32H = pstVdaData->rect[i].fHeigh;
        u32Hor = pstVdaData->horizontal_count;
        u32Ver = pstVdaData->vertical_count;
        fprintf(fp, "x=%d,y=%d,w=%d,h=%d\n", u32X, u32Y, u32W, u32H);
        for (j = 0; j < u32Ver; j++) {
            for (k = 0; k < u32Hor; k++) {
                pu8Addr = (FY_U8*)(pstVdaData->start + i * u32Hor * u32Ver + j * u32Hor + k);
                fprintf(fp, "%-2d ", *pu8Addr);
            }
            fprintf(fp, "\n");
        }
    }

    fflush(fp);
    return FY_SUCCESS;
}

static FY_S32 _VDA_PrintTwoDimDomin(FY_MDConfig_Ex_Result_t* pstVdaData) {
    FY_S32 i, j, k;
    FILE* fp = stdout;
    FY_U32 rectX, rectY, rectW, rectH, curMode;

    curMode = pstVdaData->curMode;
    // fprintf(fp, "===== %s =====\n", __FUNCTION__);
    for (i = 0; i < pstVdaData->cached_number; i++) {
        rectX = pstVdaData->rect[i].fTopLeftX;
        rectY = pstVdaData->rect[i].fTopLeftY;
        rectW = pstVdaData->rect[i].fWidth;
        rectH = pstVdaData->rect[i].fHeigh;
        fprintf(fp, "curMode=%d,x=%d,y=%d,w=%d,h=%d\n", curMode, rectX, rectY, rectW, rectH);
        for (j = 0; j < pstVdaData->vertical_count; j++) {
            for (k = 0; k < pstVdaData->horizontal_count; k++) {
                if (((rectX / curMode == k || ((rectX + rectW) / curMode - 1) == k) && j >= rectY / curMode &&
                     j <= ((rectY + rectH) / curMode - 1) && rectW != 0) ||
                    ((rectY / curMode == j || ((rectY + rectH) / curMode - 1) == j) && k >= rectX / curMode &&
                     k <= ((rectX + rectW) / curMode - 1) && rectH != 0)) {
                    fprintf(fp, " *");
                } else if (j == 0 || j == (pstVdaData->vertical_count - 1) || k == 0 ||
                           k == (pstVdaData->horizontal_count - 1)) {
                    fprintf(fp, " +");
                } else {
                    fprintf(fp, "  ");
                }
            }
            fprintf(fp, "\n");
        }
    }
    fflush(fp);
    return FY_SUCCESS;
}

static FY_S32 _VDA_PrintDiffValue(FY_MDConfig_Ex_Result_t* pstVdaData) {
    FY_S32 i, j, k;
    FY_U8* pu8Addr;
    FILE* fp = stdout;
    FY_U32 offset = 0;

    fprintf(fp, "\n======================= show diff value ==========================\n");
    for (i = 0; i < pstVdaData->cached_number; i++) {
        for (j = 0; j < pstVdaData->vertical_count; j++) {
            for (k = 0; k < pstVdaData->horizontal_count; k++) {
                offset = i * pstVdaData->horizontal_count * pstVdaData->vertical_count;
                offset += j * pstVdaData->horizontal_count;
                offset += k;
                pu8Addr = (FY_U8*)(pstVdaData->pMDValue + offset);
                fprintf(fp, "%02x ", *pu8Addr);
            }
            fprintf(fp, "\n");
        }
    }

    fflush(fp);
    return FY_SUCCESS;
}

FY_VOID MPI_VDA_save_file(VDA_FILE_INFO* pOriData, VDA_FILE_INFO* pDstData, FY_U32 count) {
    FILE* pFile = FY_NULL;
    FY_S32 aa = 0;

    if (pOriData) {
        aa = access(pOriData->name, F_OK);
        if (aa == -1) {
            printf("save file enter\n");
            VDA_OPEN_FILE(pFile, pOriData->name, "w+");
            VDA_WRITE_FILE(pFile, pOriData->pAddr, pOriData->len);
            VDA_CLOSE_FILE(pFile);
        }
    }

    if (pDstData) {
        aa = access(pDstData->name, F_OK);
        if (aa == -1) {
            VDA_OPEN_FILE(pFile, pDstData->name, "w+");
            VDA_WRITE_FILE(pFile, pDstData->pAddr, pOriData->len);
            VDA_CLOSE_FILE(pFile);
        }
    }
exit:
    return;
}

static FY_S32 VDA_MD_AllocCacheBuff(FY_U32 id) {
    FY_U32 len = 0;
    FY_U32 u32Cnt, u32OutSize;

    CHECK_ID_IS_VALID(id);
    u32Cnt = g_MdCfgInfo[id].mdCfg.resultNum;
    u32OutSize = g_MdCfgInfo[id].u32OutSize;
    VDA_Alloc_Memory(g_MdCfgInfo[id].pMdStatus, u32OutSize);
    if (u32Cnt) {
        if (g_MdCfgInfo[id].mdCfg.stype != MD_FRM_DIFF_ONLY) {
            len = u32Cnt * u32OutSize;
            VDA_Alloc_Memory(g_MdCfgInfo[id].cachePool.start, len);

            len = u32Cnt * sizeof(FY_Rect_t);
            VDA_Alloc_Memory(g_MdCfgInfo[id].cachePool.rect, len);
        }

        len = u32Cnt * u32OutSize;
        VDA_Alloc_Memory(g_MdCfgInfo[id].cachePool.pMDValue, len);
    }
exit:
    return FY_SUCCESS;
}

static FY_S32 VDA_MD_FreeCacheBuff(FY_U32 id) {
    CHECK_ID_IS_VALID(id);
    VDA_Free_Memory(g_MdCfgInfo[id].pMdStatus);
    VDA_Free_Memory(g_MdCfgInfo[id].cachePool.start);
    VDA_Free_Memory(g_MdCfgInfo[id].cachePool.rect);
    VDA_Free_Memory(g_MdCfgInfo[id].cachePool.pMDValue);
    g_MdCfgInfo[id].cachePool.cached_number = 0;
    return FY_SUCCESS;
}

FY_S32 VDA_MD_CheckChnMode(FY_U32 id, FY_S32 grp, FY_S32 chn, VPSS_PIC_DATA_S* pYcMean) {
    FY_U32 u32W_L, u32H_L, u32M_L;
    VPSS_CHN_MODE_S* pChnMode = FY_NULL;
    FY_BOOL bOpenLog = FY_FALSE;
    VPSS_CHN_MODE_S stVpuChnMode;
    FY_U32 u32Ord, u32Ord_pre;

    if (pYcMean) {
        u32Ord = g_MdCfgInfo[id].u32Ord;
        u32Ord_pre = GET_REMINDER_OFFSET((u32Ord + 1), 2);
        bOpenLog = g_MdCfgInfo[id].initParam.bOpenLog;
        if (FY_MPI_VPSS_GetChnMode(grp, chn, &stVpuChnMode) != FY_SUCCESS) {
            printf("[%s]get channelMode failed!\n", __func__);
            return FY_FAILURE;
        }
        pChnMode = &stVpuChnMode;
        g_MdCfgInfo[id].Yc[u32Ord].u32DsMode = pChnMode->mainCfg.ycmeanMode;
        if (!pChnMode->mainCfg.ycmeanMode || !pChnMode->u32Width) {
            printf("[%s]err(ycmod %d,width %d)!\n", __func__, pChnMode->mainCfg.ycmeanMode, pChnMode->u32Width);
            return FY_FAILURE;
        }

        u32W_L = g_MdCfgInfo[id].Yc[u32Ord_pre].u32Width;
        u32H_L = g_MdCfgInfo[id].Yc[u32Ord_pre].u32Height;
        u32M_L = g_MdCfgInfo[id].Yc[u32Ord_pre].u32DsMode;
        if (pYcMean->u32Width != u32W_L && pYcMean->u32Height != u32H_L && pChnMode->mainCfg.ycmeanMode != u32M_L) {
            VDA_LOG(bOpenLog, "[MD-checkMode][%d]mode %d:%d,w %d:%d,h %d:%d,size %d\n", id,
                    pChnMode->mainCfg.ycmeanMode, u32M_L, pYcMean->u32Width, u32W_L, pYcMean->u32Height, u32H_L,
                    g_MdCfgInfo[id].u32OutSize);
            return FY_FAILURE;  // skip this frame
        }
    }
    return FY_SUCCESS;
}

static FY_S32 VDA_MD_CalcDiffValue(FY_U32 id, FY_U32 totalSize) {
    FY_U8 *pLastMdPtr, *pCurMdPtr;
    FY_U8 chvalue = 0;
    FY_U32 idx = 0;
    FY_U16 u16LastPos, u16CurPos;
    FY_U8* pAtr = FY_NULL;

    u16CurPos = g_MdCfgInfo[id].u32Ord;
    u16LastPos = GET_REMINDER_OFFSET((u16CurPos + 1), 2);
    pLastMdPtr = (FY_U8*)g_MdCfgInfo[id].Dyc[u16LastPos].pVirAddr;
    pCurMdPtr = (FY_U8*)g_MdCfgInfo[id].Dyc[u16CurPos].pVirAddr;
    if (!pLastMdPtr || !pCurMdPtr) {
        return FY_FAILURE;
    }
    pAtr = (FY_U8*)g_MdCfgInfo[id].Dvalue.pVirAddr;
    if (!pAtr) {
        printf("[%s]line[%d]memory is null!\n", __func__, __LINE__);
        return FY_FAILURE;
    }
    for (idx = 0; idx < totalSize; idx++) {
        if (pLastMdPtr && pCurMdPtr) {
            chvalue = (FY_U8)GetDiffValue_Byte(*(pLastMdPtr + idx), *(pCurMdPtr + idx));
        }
        pAtr[idx] = chvalue;
    }
    return FY_SUCCESS;
}

static FY_S32 VDA_MD_DealYcMean(DEAL_YC_PARAM* pAtr) {
    FY_U32 i, j, k, cnt = 0;
    FY_U32 disW, disH, reminderW, reminderH, horNum, verNum;
    FY_U32 temp_w, temp_h;
    VDA_MD_Macro_s mdCalcParam;
    FY_U32 tmpSize = 0;
    FY_U16 mode;
    FY_VOID* pInYmean = FY_NULL;
    FY_U8* pOutYmean = FY_NULL;
    FY_U32 u32threshold;

    VDA_CHECK_PTR_NULL(pAtr);
    VDA_CHECK_PTR_NULL(pAtr->pYcApr);
    VDA_CHECK_PTR_NULL(pAtr->pOutput);

    mode = pAtr->u16OutMode;
    pInYmean = pAtr->pYcApr;
    pOutYmean = pAtr->pOutput;
    u32threshold = pAtr->u32thed;
    memset(&mdCalcParam, 0, sizeof(VDA_MD_Macro_s));
    switch (mode) {
        case DS_ONE_FOURTH:
        case DS_ONE_EIGHTH:
        case DS_ONE_SIXTEENTH:
            break;
        default:
            printf("don't support this mode!(%d)\n", mode);
            return -1;
    }

    temp_w = pAtr->inRect.u32Width;
    temp_h = pAtr->inRect.u32Height;
    horNum = GET_MACRO_BLOCK_NUM(pAtr->outRect.u32Width, mode);
    verNum = GET_MACRO_BLOCK_NUM(pAtr->outRect.u32Height, mode);
    disW = GET_DIS_OFFSET(temp_w, horNum);
    disH = GET_DIS_OFFSET(temp_h, verNum);
    reminderW = GET_REMINDER_OFFSET(temp_w, horNum);
    reminderH = GET_REMINDER_OFFSET(temp_h, verNum);
    for (i = 0; i < horNum; i++) {
        for (j = 0; j < verNum; j++) {
            FY_U32 tempX, tempY, offsetX, offsetY, tempOffset;
            tempX = disW;
            tempY = disH;
            offsetX = reminderW;
            offsetY = reminderH;
            if (i < reminderW) {
                tempX += 1;
                offsetX = i;
            }
            if (j < reminderH) {
                tempY += 1;
                offsetY = j;
            }
            tmpSize = tempX * tempY;
            VDA_CHECK_SIZE_IS_ENQ(tmpSize, &mdCalcParam.size, (FY_VOID**)&mdCalcParam.pValue);
            for (k = 0; k < tempY; k++) {
                tempOffset = j * disH + offsetY + k;
                memcpy(mdCalcParam.pValue + k * tempX, pInYmean + i * disW + offsetX + tempOffset * temp_w, tempX);
            }
            mdCalcParam.uValue = 0;
            u32threshold = MAX_VALUE(pAtr->u32thed, COMP_VALUE);
            for (k = 0; k < mdCalcParam.size; k++) {
                if (mdCalcParam.pValue[k] > u32threshold) {
                    mdCalcParam.uValue += mdCalcParam.pValue[k];
                }
            }
            cnt = i + j * horNum;
            if (mdCalcParam.size) mdCalcParam.uValue = mdCalcParam.uValue / mdCalcParam.size;
            *(pOutYmean + cnt) = (FY_U8)mdCalcParam.uValue;
        }
    }
    VDA_Free_Memory(mdCalcParam.pValue);
    return FY_SUCCESS;
}

FY_S32 VDA_MD_GetRectInfo(FY_U32 id, FY_U32 hor, FY_U32 ver, FY_U8* pResult, FY_Rect_t* pRect) {
    FY_U32 minX, maxX;
    FY_U32 minY, maxY;
    FY_U32 i, j, k;
    FY_U32 mb_size;

    CHECK_ID_IS_VALID(id);
    minX = maxX = minY = maxY = -1;
    if (pResult) {
        for (i = 0; i < ver; i++) {
            for (j = 0; j < hor; j++) {
                k = i * hor + j;
                if (*(pResult + k) == 1) {
                    if (minX == -1) {
                        minX = maxX = j;
                        minY = maxY = i;
                    } else {
                        minX = minX < j ? minX : j;
                        minY = minY < i ? minY : i;
                        maxX = maxX > j ? maxX : j;
                        maxY = maxY > i ? maxY : i;
                    }
                }
            }
        }
        if (minX != -1) {
            maxX += 1;
        }
        if (minY != -1) {
            maxY += 1;
        }
        minX = (minX == -1) ? 0 : minX;
        minY = (minY == -1) ? 0 : minY;
        maxX = (maxX == -1) ? 0 : maxX;
        maxY = (maxY == -1) ? 0 : maxY;
        // printf("minX=%d,y=%d,maxX=%d,y=%d\n",minX,minY,maxX,maxY);
        mb_size = g_MdCfgInfo[id].initParam.outputMode;
        pRect->fTopLeftX = minX * mb_size;
        pRect->fTopLeftY = minY * mb_size;
        pRect->fWidth = (maxX - minX) * mb_size;
        pRect->fHeigh = (maxY - minY) * mb_size;
    }
    return FY_SUCCESS;
}

FY_S32 VDA_MD_SetCfgInfo(FY_U32 id, FY_BOOL bEnable) {
    FY_S32 ret = FY_FAILURE;
    FY_U32 u32YcmeanMode;
    FY_BOOL bLog = FY_FALSE;
    FY_U32 u32H, u32W;
    FY_U32 u32HorMac, u32VerMac;

    // pthread_mutex_lock(&g_MdCfgInfo[id].mdMutex);
    bLog = g_MdCfgInfo[id].initParam.bOpenLog;
    u32YcmeanMode = g_MdCfgInfo[id].initParam.outputMode;
    VDA_LOG(bLog, "[%s][%d]ycmode(%d)!\n", __func__, id, u32YcmeanMode);
    u32W = g_MdCfgInfo[id].initParam.mdSize.u32Width;
    u32H = g_MdCfgInfo[id].initParam.mdSize.u32Height;
    switch (u32YcmeanMode) {
        case DS_ONE_FOURTH:
        case DS_ONE_EIGHTH:
        case DS_ONE_SIXTEENTH:
            u32W = GET_BITS_ALIGNMENT(u32W, u32YcmeanMode);
            u32H = GET_BITS_ALIGNMENT(u32H, u32YcmeanMode);
            u32HorMac = GET_DIS_OFFSET(u32W, u32YcmeanMode);
            u32VerMac = GET_DIS_OFFSET(u32H, u32YcmeanMode);
            g_MdCfgInfo[id].initParam.mdSize.u32Width = u32W;
            g_MdCfgInfo[id].initParam.mdSize.u32Height = u32H;
            g_MdCfgInfo[id].u32OutSize = u32HorMac * u32VerMac;
            g_MdCfgInfo[id].uHorMacNum = u32HorMac;
            g_MdCfgInfo[id].uVerMacNum = u32VerMac;
            break;
        default:
            printf("[%s]not support ycmode(%d)!\n", __func__, u32YcmeanMode);
            goto exit;
    }
    VDA_LOG(bLog, "[%d]allign w[%d]:h[%d]\n", id, u32W, u32H);

    if (!bEnable) {
        VDA_MD_FreeCacheBuff(id);
    } else {
        VDA_MD_AllocCacheBuff(id);
    }
exit:
    // pthread_mutex_unlock(&g_MdCfgInfo[id].mdMutex);
    return ret;
}

FY_S32 VDA_MD_SaveDebugFile(FY_U32 id, FY_U32 order, FY_U32 uSize) {
#ifdef DEBUG_SAVE_FILE
    {
        VDA_FILE_INFO oriFile, dstFile;

        memset(&oriFile, 0, sizeof(VDA_FILE_INFO));
        memset(&dstFile, 0, sizeof(VDA_FILE_INFO));

        oriFile.pAddr = g_MdCfgInfo[id].Dyc.pVirAddr;
        oriFile.len = uSize;
        sprintf(oriFile.name, "./channel%d_yc.yuv", id);

        dstFile.pAddr = (FY_CHAR*)g_MdCfgInfo[id].Dvalue.pVirAddr;
        dstFile.len = g_MdCfgInfo[id].u32OutSize;
        sprintf(dstFile.name, "./convert%d_yc.yuv", id);
        MPI_VDA_save_file(&oriFile, &dstFile, 0);
    }
#endif
    return FY_SUCCESS;
}

FY_S32 VDA_MD_motion_process(FY_U32 id, VPSS_PIC_DATA_S* pYmean) {
    FY_U32 u32Size, u32DataLen;
    FY_U32 ret = FY_SUCCESS;
    DEAL_YC_PARAM cc;
    IOE_PARAM_T hw;
    FY_U16 u32Ord, u32Ord_pre;
    FY_U32 u32D_W, u32D_H;

    VDA_CHECK_PTR_NULL(pYmean);

    u32Ord = g_MdCfgInfo[id].u32Ord;
    u32Ord_pre = GET_REMINDER_OFFSET((u32Ord + 1), 2);
    memset(&cc, 0, sizeof(cc));
    memset(&hw, 0, sizeof(hw));
    u32Size = pYmean->u32Height * pYmean->u32Width;
    u32D_W = g_MdCfgInfo[id].initParam.mdSize.u32Width / g_MdCfgInfo[id].initParam.outputMode;
    u32D_H = g_MdCfgInfo[id].initParam.mdSize.u32Height / g_MdCfgInfo[id].initParam.outputMode;
    g_MdCfgInfo[id].Dyc[u32Ord].u32Width = u32D_W;
    g_MdCfgInfo[id].Dyc[u32Ord].u32Height = u32D_H;
    VDA_Malloc_Memz(id, MD_MMZ_DYC, g_MdCfgInfo[id].u32OutSize);
    VDA_Malloc_Memz(id, MD_MMZ_OUTPUT, g_MdCfgInfo[id].u32OutSize);

    u32DataLen = g_MdCfgInfo[id].Yc[u32Ord_pre].u32DataLen;
    if (u32DataLen != u32Size) {
        printf("[%s]line[%d]size %d:%d\n", __func__, __LINE__, u32DataLen, u32Size);
        goto exit;
    }
    u32DataLen = g_MdCfgInfo[id].Yc[u32Ord].u32DataLen;
    if (u32DataLen != u32Size) {
        printf("[%s]line[%d]size %d:%d\n", __func__, __LINE__, u32DataLen, u32Size);
        goto exit;
    }
    if (g_VdaHwInfo.bOpen) {
        hw.img_y_only = 1;
        hw.img_yuv420 = 0;
        hw.img_uv_order = 0;
        hw.md_y_threshold = g_MdCfgInfo[id].mdCfg.threshold;
        hw.md_u_threshold = 0;
        hw.md_v_threshold = 0;
        hw.img_src_size_w = pYmean->u32Width;
        hw.img_src_size_h = pYmean->u32Height;
        hw.img_src_pitch_w = BIT_ALLIGN_BYTES(hw.img_src_size_w, 3, 8);
        hw.img0_base_addr_y = g_MdCfgInfo[id].Yc[u32Ord_pre].u32PhyAddr;
        hw.img0_base_addr_c = g_MdCfgInfo[id].Yc[u32Ord_pre].u32PhyAddr + u32Size;
        hw.img1_base_addr_y = g_MdCfgInfo[id].Yc[u32Ord].u32PhyAddr;
        hw.img1_base_addr_c = g_MdCfgInfo[id].Yc[u32Ord].u32PhyAddr + u32Size;
        hw.img_des_size_w = u32D_W;
        hw.img_des_size_h = u32D_H;
        hw.pixel_num_threshold = 0;
        hw.img_des_base_addr_y = g_MdCfgInfo[id].Dvalue.u32PhyAddr;
        hw.img_des_base_addr_c = g_MdCfgInfo[id].Dvalue.u32PhyAddr + hw.img_des_size_w * hw.img_des_size_h;
        hw.timeout = 500;
        ret = FY_IOE_OPERATE(&hw);
    } else {
        cc.inRect.u32Width = pYmean->u32Width;
        cc.inRect.u32Height = pYmean->u32Height;
        cc.u16OutMode = g_MdCfgInfo[id].initParam.outputMode;
        cc.outRect.u32Width = g_MdCfgInfo[id].initParam.mdSize.u32Width;
        cc.outRect.u32Height = g_MdCfgInfo[id].initParam.mdSize.u32Height;
        cc.pYcApr = g_MdCfgInfo[id].Yc[u32Ord].pVirAddr;
        cc.pOutput = g_MdCfgInfo[id].Dyc[u32Ord].pVirAddr;
        cc.u32thed = g_MdCfgInfo[id].mdCfg.threshold;
        ret = VDA_MD_DealYcMean(&cc);
        if (ret != 0) {
            ret = FY_FAILURE;
            goto exit;
        }
        u32Size = g_MdCfgInfo[id].u32OutSize;
        ret = VDA_MD_CalcDiffValue(id, u32Size);
        if (ret != 0) {
            ret = FY_FAILURE;
            goto exit;
        }
    }
exit:
    return ret;
}

FY_S32 VDA_MD_Filter(FY_U32 id) {
    FY_S32 ret = FY_SUCCESS;
    FY_U32 i, stype;
    FY_U8* pAtr = FY_NULL;
    FY_BOOL bInit, bOpenLog;

    bInit = g_MdCfgInfo[id].bInit;
    stype = g_MdCfgInfo[id].mdCfg.stype;
    bOpenLog = g_MdCfgInfo[id].initParam.bOpenLog;
    VDA_LOG(bOpenLog, "[%s]id[%d]init[%d]type[%d][%d]\n", __func__, id, bInit, stype, bOpenLog);
    if (bInit) {
        if (stype != MD_FRM_DIFF_ONLY) {
            pAtr = (FY_U8*)g_MdCfgInfo[id].Dvalue.pVirAddr;
            for (i = 0; i < g_MdCfgInfo[id].u32OutSize; i++) {
                if (pAtr && pAtr[i] * 6 > g_MdCfgInfo[id].mdCfg.threshold)
                    g_MdCfgInfo[id].pMdStatus[i] = 1;
                else
                    g_MdCfgInfo[id].pMdStatus[i] = 0;
            }
        }
    }
    return ret;
}

FY_S32 VDA_CD_CheckDistEnable(FY_S32 VpssGrp) {
    FY_S32 result = FY_SUCCESS;
    VPSS_GRP_ATTR_S grp_info;

    result = FY_MPI_VPSS_GetGrpAttr(VpssGrp, &grp_info);
    if (!grp_info.bHistEn) {
        result = FY_FAILURE;
    }
    return result;
}

static FY_S32 VDA_CD_ProcessDiffValue(FY_U32 id, FY_U8* pSrc, FY_U32 size) {
    FY_S32 i;
    FY_U8 uVle;
    FY_VDA_CD_CFG* pAtr = FY_NULL;
    FY_U32 u32Level = 0;
    FY_U32 u32Pct = 100, u32MinV, u32MaxV;
    FY_BOOL bOpenLog = FY_FALSE;
    FY_U32 uDcfg = 0;
    FY_U32 u32MinPct = 0, u32MinValue = 0, u32MinCnt = 0;

    if (!pSrc || !size) {
        return FY_FAILURE;
    }
    pAtr = &g_CoverCfgInfo[id];
    bOpenLog = pAtr->initParam.bOpenLog;
    uDcfg = g_CoverCfgInfo[id].cdCfg.changevalue;
    memset(&pAtr->count[0], 0, MAX_CNT * sizeof(FY_U32));
    for (i = 0; i < size; i++) {
        uVle = pSrc[i];
        if (pAtr) {
            pAtr->count[uVle]++;
        }
    }

    u32Level = pAtr->cdCfg.level;
    switch (u32Level) {
        case CD_LEVEL_HIGH:
            u32MinV = 32;
            u32MaxV = 112;
            u32Pct = 96;
            break;
        case CD_LEVEL_MID:
            u32MinV = 32;
            u32MaxV = 112;
            u32Pct = 98;
            break;
        case CD_LEVEL_LOW:
            u32MinV = 32;
            u32MaxV = 112;
            u32Pct = 99;
            break;
        default:
            break;
    }

    if (u32Pct) {
        u32MinPct = u32Pct;
        if (uDcfg > 0 && uDcfg < 40) {
            u32Pct -= uDcfg;
        }
        u32Level = 0;
        for (i = u32MinV; i < u32MaxV; i++) {
            u32Level += pAtr->count[i];
        }
        for (i = 0; i < u32MinV; i++) {
            u32MinCnt += pAtr->count[i];
            u32MinValue += i * pAtr->count[i];
        }
        u32Level = u32Level * 100 / size;
        if ((u32Level >= u32Pct) || ((u32MinCnt * 100 > u32MinPct * size) && (u32MinValue < size))) {
            g_CoverCfgInfo[id].cdResult = FY_TRUE;
        } else {
            g_CoverCfgInfo[id].cdResult = FY_FALSE;
        }
        VDA_LOG(bOpenLog, "[%s]id[%d],lv[%d],percent[%d],[%d/%d]\n", __func__, id, u32Level, u32Pct, u32MinValue,
                u32MinCnt);
    }

    return FY_SUCCESS;
}

FY_S32 VDA_CD_process(FY_S32 id) {
    FY_S32 VpssGrp;
    FY_S32 result = FY_SUCCESS;
    VPSS_HIST_STAT_S stHist;
    FY_U32 i = 0, u32Cnt, u32Sub;
    FY_U32 u32TotalCnt = 0, u32MM = 0, u32Level;
    FY_VDA_CD_CFG* pAtr = FY_NULL;
    FY_U32 u32Min = 0, u32Max = 19, u32Pct = 100, u32pp = 0;
    FY_BOOL bOpenLog = FY_FALSE;

    pAtr = &g_CoverCfgInfo[id];
    VpssGrp = g_CoverCfgInfo[id].initParam.VpssGrp;
    bOpenLog = pAtr->initParam.bOpenLog;
    result = VDA_CD_CheckDistEnable(VpssGrp);
    if (result != FY_SUCCESS) {
        if (pAtr->u32FailCnt < 1) {
            printf("[%s][CD%d]grpID-%d result=%x\n", __func__, id, VpssGrp, result);
        }
        pAtr->u32FailCnt++;
        goto exit;
    }

    memset(&stHist, 0, sizeof(VPSS_HIST_STAT_S));
    result = FY_MPI_VPSS_GetHistStat(VpssGrp, &stHist);
    if (result != FY_SUCCESS) {
        if (pAtr->u32FailCnt < 3) {
            printf("[%s][%d][CD%d]result=%x\n", __func__, __LINE__, id, result);
        }
        pAtr->u32FailCnt++;
        goto exit;
    }
    u32Level = pAtr->cdCfg.level;
    switch (u32Level) {
        case CD_LEVEL_HIGH:
            u32Min = 10;
            u32Pct = 99;
            break;
        case CD_LEVEL_MID:
            u32Min = 13;
            u32Pct = 98;
            break;
        case CD_LEVEL_LOW:
            u32Min = 15;
            u32Pct = 93;
            break;
        default:
            break;
    }

    for (i = 0; i < 33; i++) {
        u32Cnt = stHist.histBin[i][0];
        u32Sub = stHist.histBin[i][1];
        if (u32Sub) {
        }
        if (i > u32Min && i < u32Max) {
            u32MM += u32Cnt;
        }
        u32TotalCnt += u32Cnt;
    }
    if (u32TotalCnt) {
        u32pp = (u32MM * 100 / u32TotalCnt);
        if (u32pp >= u32Pct) {
            g_CoverCfgInfo[id].cdResult = FY_TRUE;
        } else {
            g_CoverCfgInfo[id].cdResult = FY_FALSE;
        }
    }
    pAtr->u32FailCnt = 0;
    VDA_LOG(bOpenLog, "SHOW CD%d T[%d] [%d]%% ? [%d],l[%d]\n", id, u32TotalCnt, u32pp, u32Pct, u32Level);
exit:
    return result;
}

FY_VOID* VDA_Thread_Loop(FY_VOID* pdata) {
    FY_U32 i = 0;
    FY_S32 result;
    VDA_THREAD_S* pAttr = (VDA_THREAD_S*)pdata;
    FY_CHAR name[20] = "\0";
    FY_U16 u16_time;

    if (!pdata) {
        goto exit;
    }
    sprintf(name, "MPI_VDA_%d", pAttr->u32Type);
    prctl(PR_SET_NAME, name);
    while (pAttr->bStart) {
        for (i = 0; i < MAX_MD_CHANNEL_NUM; i++) {
            if (pAttr->u32Type == VDA_TYPE_MD) {
                if (g_MdCfgInfo[i].bInit) {
                    result = MPI_VDA_MD_CD_Check(VDA_TYPE_MD, i);
                    if (result == FY_SUCCESS) {
                        // MPI_VDA_MD_Print(i,2);
                    }
                }
                u16_time = 160 / g_MdCfgInfo[i].initParam.maxChnNum;
                usleep(u16_time * 1000);
            } else {
                if (g_CoverCfgInfo[i].bInit) {
                    result = MPI_VDA_MD_CD_Check(VDA_TYPE_CD, i);
                    if (result == FY_SUCCESS) {
                    }
                }
            }
        }
        usleep(30 * 1000);
    }
exit:
    return FY_NULL;
}

FY_S32 VDA_Create_thread(FY_U32 type, FY_U16 prior) {
    pthread_attr_t attr;
    FY_S32 policy, inher;
    struct sched_param param;
    FY_S32 ret;

    if (g_VdaThrInfo[type].t_id == 0) {
        pthread_attr_init(&attr);
        ret = pthread_attr_getinheritsched(&attr, &inher);
        if (ret != 0) {
            printf("[%s]line[%d][%d] %s\n", __func__, __LINE__, type, strerror(ret));
            return FY_FAILURE;
        }
        if (inher == PTHREAD_EXPLICIT_SCHED) {
        } else if (inher == PTHREAD_INHERIT_SCHED) {
            inher = PTHREAD_EXPLICIT_SCHED;
        }
        ret = pthread_attr_setinheritsched(&attr, inher);
        if (ret != 0) {
            return FY_FAILURE;
        }

        policy = SCHED_FIFO;
        ret = pthread_attr_setschedpolicy(&attr, policy);
        if (ret != 0) {
            printf("[%s]line[%d][%d] %s\n", __func__, __LINE__, type, strerror(ret));
            return FY_FAILURE;
        }
        param.sched_priority = prior;
        ret = pthread_attr_setschedparam(&attr, &param);
        if (ret != 0) {
            printf("[%s]line[%d][%d] %s\n", __func__, __LINE__, type, strerror(ret));
            return FY_FAILURE;
        }
        g_VdaThrInfo[type].bStart = FY_TRUE;
        g_VdaThrInfo[type].u32Type = type;
        pthread_create(&g_VdaThrInfo[type].t_id, &attr, VDA_Thread_Loop, &g_VdaThrInfo[type]);
    }
    return FY_SUCCESS;
}

FY_S32 MPI_VDA_GetYcmeanData(FY_VDA_TYPE type, FY_U32 id, FY_VOID* pYcmean) {
    FY_S32 errs = FY_SUCCESS;
    FY_VOID* pAddr = FY_NULL;
    FY_S32 VpssGrp, VpssChn;
    FY_U32 u32Size_yc = 0, u32Size_md = 0;
    VPSS_PIC_DATA_S ycmean;
    FY_BOOL bLog = FY_FALSE;
    FY_U32 u32Ord = 0;

    VDA_CHECK_PTR_NULL(pYcmean);
    if (type == VDA_TYPE_MD) {
        VpssGrp = g_MdCfgInfo[id].initParam.VpssGrp;
        VpssChn = g_MdCfgInfo[id].initParam.VpssChn;
        bLog = g_MdCfgInfo[id].initParam.bOpenLog;
        u32Ord = g_MdCfgInfo[id].u32Ord;
    } else {
        u32Size_md = g_CoverCfgInfo[id].Yc.u32DataLen;
        VpssGrp = g_CoverCfgInfo[id].initParam.VpssGrp;
        VpssChn = g_CoverCfgInfo[id].initParam.VpssChn;
        bLog = g_CoverCfgInfo[id].initParam.bOpenLog;
    }

    errs = FY_MPI_VPSS_GetYcMean(VpssGrp, VpssChn, &ycmean);
    if (errs != FY_SUCCESS) {
        goto exit;
    }
    VDA_LOG(bLog, "[%s][T%d:%d]phy=0x%x,yc w[%d]h[%d],md_size=%d\n", __func__, type, id, ycmean.u32PhyAddr,
            ycmean.u32Width, ycmean.u32Height, u32Size_md);
    memcpy(pYcmean, &ycmean, sizeof(ycmean));
    u32Size_yc = ycmean.u32Width * ycmean.u32Height;
    if (type == VDA_TYPE_MD) {
        VDA_Malloc_Memz(id, MD_MMZ_YC, u32Size_yc);
        pAddr = g_MdCfgInfo[id].Yc[u32Ord].pVirAddr;
        g_MdCfgInfo[id].Yc[u32Ord].u32Width = ycmean.u32Width;
        g_MdCfgInfo[id].Yc[u32Ord].u32Height = ycmean.u32Height;
    } else {
        pAddr = g_CoverCfgInfo[id].Yc.pVirAddr;
        VDA_CHECK_SIZE_IS_ENQ(u32Size_yc, &u32Size_md, &pAddr);
        g_CoverCfgInfo[id].Yc.u32DataLen = u32Size_md;
        g_CoverCfgInfo[id].Yc.pVirAddr = pAddr;
    }
    ycmean.pVirAddr = (void*)FY_MPI_SYS_Mmap(ycmean.u32PhyAddr, u32Size_yc);
    memcpy(pAddr, ycmean.pVirAddr, u32Size_yc);
    FY_MPI_SYS_Munmap(ycmean.pVirAddr, u32Size_yc);
    FY_MPI_VPSS_ReleasePicData(VpssGrp, VpssChn, &ycmean);
exit:
    VDA_LOG(bLog, "[%s]exit[T%d:%d]u32Ord[%d]yc_size=%d\n", __func__, type, id, u32Ord, u32Size_yc);
    return errs;
}

FY_S32 MPI_VDA_MD_CD_Check(FY_VDA_TYPE type, FY_S32 id) {
    FY_BOOL bLog;
    FY_S32 frameDiff;
    FY_S32 time_delay;
    FY_S32 VpssGrp, VpssChn;
    FY_U32 ret = -1;
    FY_U32 u32Size;
    FY_U32 u32Ord = 0;
    FY_U64 u64TmpVe;
    FY_U8* pAtr = FY_NULL;
    VPSS_PIC_DATA_S ycmean;

    CHECK_ID_IS_VALID(id);

    memset(&ycmean, 0, sizeof(VPSS_PIC_DATA_S));
    ycmean.milliSec = 200;
    switch (type) {
        case VDA_TYPE_MD: {
            pthread_mutex_lock(&g_MdCfgInfo[id].mdMutex);
            if (g_MdCfgInfo[id].mdCfg.enable) {
                u32Ord = g_MdCfgInfo[id].u32Ord;
                bLog = g_MdCfgInfo[id].initParam.bOpenLog;
                VpssGrp = g_MdCfgInfo[id].initParam.VpssGrp;
                VpssChn = g_MdCfgInfo[id].initParam.VpssChn;
#ifdef DEBUG_READ_YC_FILE
                {
                    FILE* pYcFile = FY_NULL;
                    FY_U32 fileLen = 0, readLen;
                    FY_U32 vbBlk, u32PhyAddr, u32Size = 65280;
                    FY_U8* pVirAddr = FY_NULL;

                    if (g_readFile_flag)
                        VDA_OPEN_FILE(pYcFile, TEST_YC_FILE_NAME1, "r");
                    else
                        VDA_OPEN_FILE(pYcFile, TEST_YC_FILE_NAME2, "r");
                    fseek(pYcFile, 0, SEEK_END);
                    fileLen = ftell(pYcFile);
                    u32Size = fileLen;
                    vbBlk = FY_MPI_VB_GetBlock(VB_INVALID_POOLID, u32Size, NULL);
                    if (VB_INVALID_HANDLE == vbBlk) {
                        printf("\033[32m[MD]FY_MPI_VB_GetBlock err!\033[0m\n");
                    }
                    u32PhyAddr = FY_MPI_VB_Handle2PhysAddr(vbBlk);
                    if (0 != u32PhyAddr) {
                        pVirAddr = (FY_U8*)FY_MPI_SYS_Mmap(u32PhyAddr, u32Size);
                    }
                    pAtr = g_MdCfgInfo[id].Yc[u32Ord].pVirAddr;
                    u32Size = g_MdCfgInfo[id].Yc[u32Ord].u32DataLen;
                    VDA_CHECK_SIZE_IS_ENQ(u32Size, &u32Size, (FY_VOID**)&pAtr);
                    fseek(pYcFile, 0, SEEK_SET);
                    VDA_READ_FILE(pYcFile, pAtr, fileLen, readLen);
                    VDA_CLOSE_FILE(pYcFile);
                    g_readFile_flag = !g_readFile_flag;
                    frameDiff = g_MdCfgInfo[id].uFrmDelay + 1;
                    if (0 != u32PhyAddr) {
                        FY_MPI_SYS_Munmap(pVirAddr, u32Size);
                        FY_MPI_VB_ReleaseBlock(vbBlk);
                    }
                    g_MdCfgInfo[id].Yc[u32Ord].pVirAddr = pAtr;
                    g_MdCfgInfo[id].Yc[u32Ord].u32DataLen = u32Size;
                }
#else
                {
                    VDA_LOG(bLog, "[%s][%d] enter\n", __func__, id);
                    ret = MPI_VDA_GetYcmeanData(type, id, &ycmean);
                    if (ret != FY_SUCCESS) {
                        VDA_LOG(bLog, "[%s][%d]get ycmean error(0x%x)\n", __func__, id, ret);
                        goto exit;
                    }
                    if (g_MdCfgInfo[id].mdCfg.checkMode != VDA_CHECK_TIME) {
                        g_MdCfgInfo[id].u64CurPts = ycmean.u32TimeRef;
                        u64TmpVe = g_MdCfgInfo[id].u64CurPts - g_MdCfgInfo[id].u64LstPts;
                        time_delay = MAX_VALUE(u64TmpVe, 0);
                        frameDiff = time_delay;
                    } else {
                        g_MdCfgInfo[id].u64CurPts = ycmean.u64pts;
                        u64TmpVe = g_MdCfgInfo[id].u64CurPts - g_MdCfgInfo[id].u64LstPts;
                        time_delay = MAX_VALUE(u64TmpVe, 0);
                        frameDiff = time_delay / 1000;
                    }
                    g_MdCfgInfo[id].uFrmDelay = MAX_VALUE(g_MdCfgInfo[id].mdCfg.framedelay, 0);

                    VDA_LOG(bLog, "[%s]MD[id%d]ord[%d]diff[%d]?=[%d],pts[%lld][%lld]\n", __func__, id, u32Ord,
                            frameDiff, g_MdCfgInfo[id].uFrmDelay, g_MdCfgInfo[id].u64CurPts, g_MdCfgInfo[id].u64LstPts);

#ifdef DEBUG_READ_YC_FILE
#else
                    ret = VDA_MD_CheckChnMode(id, VpssGrp, VpssChn, &ycmean);
                    if (ret != FY_SUCCESS) {
                        VDA_LOG(bLog, "[MD][%d]drop 1st[%d],w/h[%d:%d]\n", id, u32Ord, ycmean.u32Width,
                                ycmean.u32Height);
                        printf("[MD][%d]drop 1st[%d],w/h[%d:%d]\n", id, u32Ord, ycmean.u32Width, ycmean.u32Height);
                        goto exit;
                    }
#endif

                    pAtr = g_MdCfgInfo[id].Dvalue.pVirAddr;
                    u32Size = g_MdCfgInfo[id].u32OutSize;
                    if (pAtr && u32Size) {
                        memset(pAtr, 0, u32Size);
                    }
                }
#endif
                if (frameDiff >= g_MdCfgInfo[id].uFrmDelay) {
                    g_MdCfgInfo[id].u64LstPts = g_MdCfgInfo[id].u64CurPts;
                } else {
                    ret = FY_FAILURE;
                    goto exit;
                }

                ret = VDA_MD_motion_process(id, &ycmean);
                if (ret != FY_SUCCESS) {
                    goto exit;
                }

                VDA_MD_SaveDebugFile(id, u32Ord, u32Size);
                VDA_MD_Filter(id);
                MPI_VDA_MD_CacheResult(id);
            }
        } break;

        case VDA_TYPE_CD: {
            pthread_mutex_lock(&g_CoverCfgInfo[id].cdMutex);
            if (g_CoverCfgInfo[id].cdCfg.enable) {
                bLog = g_CoverCfgInfo[id].initParam.bOpenLog;

                ret = MPI_VDA_GetYcmeanData(type, id, &ycmean);
                if (ret != FY_SUCCESS) {
                    goto exit;
                }

                if (g_CoverCfgInfo[id].cdCfg.checkMode != VDA_CHECK_TIME) {
                    g_CoverCfgInfo[id].u64CurPts = ycmean.u32TimeRef;
                    u64TmpVe = g_CoverCfgInfo[id].u64CurPts - g_CoverCfgInfo[id].u64LstPts;
                    time_delay = MAX_VALUE(u64TmpVe, 0);
                    frameDiff = time_delay;
                } else {
                    g_CoverCfgInfo[id].u64CurPts = ycmean.u64pts;
                    u64TmpVe = g_CoverCfgInfo[id].u64CurPts - g_CoverCfgInfo[id].u64LstPts;
                    time_delay = MAX_VALUE(u64TmpVe, 0);
                    frameDiff = time_delay / 1000;
                }
                g_CoverCfgInfo[id].uFrmDelay = MAX_VALUE(g_CoverCfgInfo[id].cdCfg.framedelay, 0);

                VDA_LOG(bLog, "[CD%d]time_delay=%d,frameDiff=%d -- %d\n", id, time_delay, frameDiff,
                        g_CoverCfgInfo[id].uFrmDelay);
                if (frameDiff >= g_CoverCfgInfo[id].uFrmDelay) {
                    g_CoverCfgInfo[id].u64LstPts = g_CoverCfgInfo[id].u64CurPts;
                } else {
                    ret = FY_FAILURE;
                    goto exit;
                }
                pAtr = g_CoverCfgInfo[id].Yc.pVirAddr;
                u32Size = g_CoverCfgInfo[id].Yc.u32DataLen;
                VDA_LOG(bLog, "[CD%d] size[%d] pAtr[%p]\n", id, u32Size, pAtr);
                if (pAtr) {
                    VDA_CD_ProcessDiffValue(id, pAtr, u32Size);
                }
                VDA_CD_process(id);
            }
        } break;
    }
    ret = FY_SUCCESS;
exit:
    switch (type) {
        case VDA_TYPE_MD: {
            g_MdCfgInfo[id].u32Ord = GET_REMINDER_OFFSET((u32Ord + 1), 2);
            pthread_mutex_unlock(&g_MdCfgInfo[id].mdMutex);
        } break;

        case VDA_TYPE_CD: {
            pthread_mutex_unlock(&g_CoverCfgInfo[id].cdMutex);
        } break;
    }
    return ret;
}

FY_S32 MPI_VDA_MD_Init(FY_VDA_INIT_cfg* pCfg) {
    FY_U32 id = 0, i;
    FY_U32 type = VDA_TYPE_MD;

    VDA_CHECK_PTR_NULL(pCfg);
    id = pCfg->ID;
    CHECK_ID_IS_VALID(id);

    if (pCfg->maxChnNum <= 0 || pCfg->maxChnNum > MAX_MD_CHANNEL_NUM) {
        pCfg->maxChnNum = MAX_MD_CHANNEL_NUM;
    }

    if (id > pCfg->maxChnNum) {
        printf("[MPI_VDA_MD_Init]failed number error\n");
        return FY_FAILURE;
    }

    if (g_MdCfgInfo[id].bInit) {
        printf("[MPI_VDA_MD_Init]has inited!\n");
        return FY_SUCCESS;
    } else {
        memset(&g_MdCfgInfo[id], 0, sizeof(FY_VDA_MD_CFG));
        if (pthread_mutex_init(&g_MdCfgInfo[id].mdMutex, 0)) {
            return FY_FAILURE;
        }
    }

    show_MD_VERSION(id);

    pCfg->VpssChn = VPSS_CHN0;
    memcpy(&g_MdCfgInfo[id].initParam, pCfg, sizeof(FY_VDA_INIT_cfg));

    if (pCfg->threadMode == DRV_THREAD_MODE) {
        VDA_Create_thread(type, 75);
    } else {
        if (g_VdaThrInfo[type].t_id != 0) {
            g_VdaThrInfo[type].bStart = FY_FALSE;
            pthread_join(g_VdaThrInfo[type].t_id, 0);
            g_VdaThrInfo[type].t_id = 0;
        }
    }
    pthread_mutex_lock(&g_MdCfgInfo[id].mdMutex);
    g_MdCfgInfo[id].bInit = FY_TRUE;
    for (i = 0; i < CHECK_BUFFER_SIZE; i++) {
        sprintf(g_MdCfgInfo[id].Yc[i].name, "VDA-MD-Yc%d-%d", i, id);
        sprintf(g_MdCfgInfo[id].Dyc[i].name, "VDA-MD-Dyc%d-%d", i, id);
    }
    sprintf(g_MdCfgInfo[id].Dvalue.name, "VDA-MD-Dvalue-%d", id);
    pthread_mutex_unlock(&g_MdCfgInfo[id].mdMutex);
    return FY_SUCCESS;
}

FY_S32 MPI_VDA_MD_DeInit(FY_U32 id) {
    int i = 0;

    CHECK_ID_IS_VALID(id);
    if (g_MdCfgInfo[id].bInit) {
        pthread_mutex_lock(&g_MdCfgInfo[id].mdMutex);
        g_MdCfgInfo[id].bInit = FY_FALSE;
        g_MdCfgInfo[id].mdCfg.enable = FY_FALSE;
        VDA_MD_FreeCacheBuff(id);
        for (i = 0; i < CHECK_BUFFER_SIZE; i++) {
            VDA_Free_Memz(id, MD_MMZ_YC, i);
            VDA_Free_Memz(id, MD_MMZ_DYC, i);
        }
        VDA_Free_Memz(id, MD_MMZ_OUTPUT, 0);
        pthread_mutex_unlock(&g_MdCfgInfo[id].mdMutex);
        pthread_mutex_destroy(&g_MdCfgInfo[id].mdMutex);
        memset(&g_MdCfgInfo[id], 0, sizeof(FY_VDA_MD_CFG));
    }
    return FY_SUCCESS;
}

FY_S32 MPI_VDA_MD_SetConfig(FY_VDA_MD_CFG_t* pMdExCfg) {
    FY_U32 id = 0;
    FY_BOOL bOpenLog = FY_FALSE;

    VDA_CHECK_PTR_NULL(pMdExCfg);
    id = pMdExCfg->ID;
    CHECK_ID_IS_VALID(id);
    VDA_MD_IS_INIT(id);

    if ((pMdExCfg->threshold > 255) && (pMdExCfg->threshold <= 0)) {
        printf("[%s]thresholderror, correct range is (0,255]!\n", __func__);
        return FY_FAILURE;
    }
    bOpenLog = g_MdCfgInfo[id].initParam.bOpenLog;
    pthread_mutex_lock(&g_MdCfgInfo[id].mdMutex);
    VDA_LOG(bOpenLog, "[%s][%d]thr %d,delay %d,en %d\n", __func__, id, pMdExCfg->threshold, pMdExCfg->framedelay,
            pMdExCfg->enable);
    VDA_MD_SetCfgInfo(id, pMdExCfg->enable);
    g_MdCfgInfo[id].mdCfg.ID = pMdExCfg->ID;
    g_MdCfgInfo[id].mdCfg.enable = pMdExCfg->enable;
    g_MdCfgInfo[id].mdCfg.framedelay = pMdExCfg->framedelay;
    g_MdCfgInfo[id].mdCfg.checkMode = pMdExCfg->checkMode;
    g_MdCfgInfo[id].mdCfg.resultNum = pMdExCfg->resultNum;
    g_MdCfgInfo[id].mdCfg.threshold = pMdExCfg->threshold;  //(255 - pMdExCfg->threshold);
    g_MdCfgInfo[id].mdCfg.stype = pMdExCfg->stype;
    VDA_LOG(bOpenLog, "[%s][%d]en %d\n", __func__, id, g_MdCfgInfo[id].mdCfg.enable);
    pthread_mutex_unlock(&g_MdCfgInfo[id].mdMutex);
    return FY_SUCCESS;
}

FY_S32 MPI_VDA_MD_GetConfig(FY_VDA_MD_CFG_t* pMdExCfg) {
    FY_U32 id = 0;

    VDA_CHECK_PTR_NULL(pMdExCfg);
    id = pMdExCfg->ID;
    CHECK_ID_IS_VALID(id);
    VDA_MD_IS_INIT(id);

    pthread_mutex_lock(&g_MdCfgInfo[id].mdMutex);
    pMdExCfg->enable = g_MdCfgInfo[id].mdCfg.enable;
    pMdExCfg->threshold = g_MdCfgInfo[id].mdCfg.threshold;
    pMdExCfg->framedelay = g_MdCfgInfo[id].mdCfg.framedelay;
    pMdExCfg->checkMode = g_MdCfgInfo[id].mdCfg.checkMode;
    pMdExCfg->resultNum = g_MdCfgInfo[id].mdCfg.resultNum;
    pMdExCfg->stype = g_MdCfgInfo[id].mdCfg.stype;
    pthread_mutex_unlock(&g_MdCfgInfo[id].mdMutex);

    return FY_SUCCESS;
}

FY_S32 MPI_VDA_MD_CacheResult(FY_U32 id) {
    FY_U32 offset = 0, size = 0, index = 0;
    FY_Rect_t rect = {0};
    FY_S32 result = FY_FAILURE;
    FY_BOOL bOpenLog = FY_FALSE;
    FY_U32 u32HorNum, u32VerNum;
    FY_U32 u32CacheNum, u32TotalNum;
    FY_VOID* pSrc = FY_NULL;
    FY_VOID* pDst = FY_NULL;

    CHECK_ID_IS_VALID(id);
    VDA_MD_IS_INIT(id);

    bOpenLog = g_MdCfgInfo[id].initParam.bOpenLog;
    u32HorNum = g_MdCfgInfo[id].uHorMacNum;
    u32VerNum = g_MdCfgInfo[id].uVerMacNum;
    VDA_LOG(bOpenLog, "[%s][id-%d]hor %d ver %d\n", __func__, id, u32HorNum, u32VerNum);

    if (g_MdCfgInfo[id].mdCfg.stype != MD_FRM_DIFF_ONLY) {
        VDA_MD_GetRectInfo(id, u32HorNum, u32VerNum, g_MdCfgInfo[id].pMdStatus, &rect);
    }

    if (!g_MdCfgInfo[id].cachePool.pMDValue) {
        VDA_LOG(bOpenLog, "[%s][%d] not set memory\n", __func__, id);
        goto exit;
    }
    u32CacheNum = g_MdCfgInfo[id].cachePool.cached_number;
    u32TotalNum = g_MdCfgInfo[id].mdCfg.resultNum;
    if (u32TotalNum <= 0) {
        goto exit;
    }
    if (rect.fWidth == 0 && rect.fHeigh == 0) {
        result = FY_SUCCESS;
        // goto exit;
    }
    VDA_LOG(bOpenLog, "[%s][%d]L %d,T %d,W %d,H %d,cacheN [%d:%d]\n", __func__, id, rect.fTopLeftX, rect.fTopLeftY,
            rect.fWidth, rect.fHeigh, u32CacheNum, u32TotalNum);
    if (u32CacheNum < u32TotalNum) {
        index = g_MdCfgInfo[id].cachePool.cached_number;

        if (g_MdCfgInfo[id].mdCfg.stype != MD_FRM_DIFF_ONLY) {
            size = g_MdCfgInfo[id].u32OutSize;
            offset = index * size;
            pSrc = (FY_VOID*)g_MdCfgInfo[id].pMdStatus;
            pDst = (FY_VOID*)&g_MdCfgInfo[id].cachePool.start[offset];
            if (pSrc && pDst) memcpy(pDst, pSrc, size);

            size = sizeof(FY_Rect_t);
            offset = index * size;
            pSrc = (FY_VOID*)&rect;
            pDst = (FY_VOID*)&g_MdCfgInfo[id].cachePool.rect[offset];
            if (pSrc && pDst) memcpy(pDst, pSrc, size);
        }

        size = g_MdCfgInfo[id].u32OutSize;
        offset = index * size;
        pSrc = (FY_VOID*)g_MdCfgInfo[id].Dvalue.pVirAddr;
        pDst = (FY_VOID*)&g_MdCfgInfo[id].cachePool.pMDValue[offset];
        if (pSrc && pDst) memcpy(pDst, pSrc, size);
        g_MdCfgInfo[id].cachePool.cached_number++;
    } else {
        if (g_MdCfgInfo[id].mdCfg.resultNum > 1) {
            g_MdCfgInfo[id].cachePool.cached_number = g_MdCfgInfo[id].mdCfg.resultNum;

            if (g_MdCfgInfo[id].mdCfg.stype != MD_FRM_DIFF_ONLY) {
                offset = g_MdCfgInfo[id].u32OutSize;
                size = (g_MdCfgInfo[id].mdCfg.resultNum - 1) * offset;
                pSrc = (FY_VOID*)g_MdCfgInfo[id].cachePool.start + offset;
                pDst = (FY_VOID*)g_MdCfgInfo[id].cachePool.start;
                if (pSrc && pDst) memcpy(pDst, pSrc, size);

                pSrc = (FY_VOID*)g_MdCfgInfo[id].pMdStatus;
                pDst = (FY_VOID*)g_MdCfgInfo[id].cachePool.start + size;
                if (pSrc && pDst) memcpy(pDst, pSrc, g_MdCfgInfo[id].u32OutSize);

                offset = sizeof(FY_Rect_t);
                size = (g_MdCfgInfo[id].mdCfg.resultNum - 1) * offset;
                pSrc = (FY_VOID*)g_MdCfgInfo[id].cachePool.rect + offset;
                pDst = (FY_VOID*)g_MdCfgInfo[id].cachePool.rect;
                if (pSrc && pDst) memcpy(pDst, pSrc, size);

                pSrc = (FY_VOID*)&rect;
                pDst = (FY_VOID*)g_MdCfgInfo[id].cachePool.rect + size;
                if (pSrc && pDst) memcpy(pDst, pSrc, offset);
            }

            offset = g_MdCfgInfo[id].u32OutSize;
            size = (g_MdCfgInfo[id].mdCfg.resultNum - 1) * offset;
            pSrc = (FY_VOID*)g_MdCfgInfo[id].cachePool.pMDValue + offset;
            pDst = (FY_VOID*)g_MdCfgInfo[id].cachePool.pMDValue;
            if (pSrc && pDst) memcpy(pDst, pSrc, size);

            pSrc = (FY_VOID*)g_MdCfgInfo[id].Dvalue.pVirAddr;
            pDst = (FY_VOID*)g_MdCfgInfo[id].cachePool.pMDValue + size;
            if (pSrc && pDst) memcpy(pDst, pSrc, offset);
        } else {
            size = g_MdCfgInfo[id].u32OutSize;

            if (g_MdCfgInfo[id].mdCfg.stype != MD_FRM_DIFF_ONLY) {
                pSrc = (FY_VOID*)g_MdCfgInfo[id].pMdStatus;
                pDst = (FY_VOID*)g_MdCfgInfo[id].cachePool.start;
                if (pSrc && pDst) memcpy(pDst, pSrc, size);

                pSrc = (FY_VOID*)&rect;
                pDst = (FY_VOID*)g_MdCfgInfo[id].cachePool.rect;
                if (pSrc && pDst) memcpy(pDst, pSrc, sizeof(FY_Rect_t));
            }
            pSrc = (FY_VOID*)g_MdCfgInfo[id].Dvalue.pVirAddr;
            pDst = (FY_VOID*)g_MdCfgInfo[id].cachePool.pMDValue;
            if (pSrc && pDst) memcpy(pDst, pSrc, size);
        }
    }
    g_MdCfgInfo[id].cachePool.time = g_MdCfgInfo[id].u64CurPts;
    g_MdCfgInfo[id].cachePool.horizontal_count = g_MdCfgInfo[id].uHorMacNum;
    g_MdCfgInfo[id].cachePool.vertical_count = g_MdCfgInfo[id].uVerMacNum;
    g_MdCfgInfo[id].cachePool.curMode = g_MdCfgInfo[id].initParam.outputMode;
    g_MdCfgInfo[id].cachePool.moveRectNum = 1;
    result = FY_SUCCESS;
exit:
    return result;
}

FY_S32 MPI_VDA_MD_GetResult(FY_U32 id, FY_MDConfig_Ex_Result_t* pMDExResult) {
    CHECK_ID_IS_VALID(id);
    VDA_MD_IS_INIT(id);
    VDA_CHECK_PTR_NULL(pMDExResult);
    pthread_mutex_lock(&g_MdCfgInfo[id].mdMutex);
    memcpy(pMDExResult, &g_MdCfgInfo[id].cachePool, sizeof(FY_MDConfig_Ex_Result_t));
    pthread_mutex_unlock(&g_MdCfgInfo[id].mdMutex);
    return FY_SUCCESS;
}

FY_VOID MPI_VDA_MD_Print(FY_U32 id, FY_U32 timeout) {
    FY_MDConfig_Ex_Result_t result;
    FY_S32 sRet;

    sRet = MPI_VDA_MD_GetResult(id, &result);
    if (sRet == FY_SUCCESS) {
        switch (timeout) {
            case 0:
                _VDA_PrintTwoDimDomin(&result);
                break;
            case 1:
                _VDA_MdPrtPoint(&result);
                break;
            case 2:
                _VDA_PrintDiffValue(&result);
                break;
            default:
                _VDA_PrintTwoDimDomin(&result);
                usleep(timeout * 1000);
                break;
        }
    }
}

FY_S32 MPI_VDA_CD_Init(FY_VDA_INIT_cfg* pCfg) {
    FY_U32 id = 0;
    FY_U32 type = VDA_TYPE_CD;

    VDA_CHECK_PTR_NULL(pCfg);
    CHECK_ID_IS_VALID(pCfg->ID);

    if (pCfg->maxChnNum <= 0 || pCfg->maxChnNum > MAX_MD_CHANNEL_NUM) {
        pCfg->maxChnNum = MAX_MD_CHANNEL_NUM;
    }
    if (pCfg->ID > pCfg->maxChnNum) {
        printf("[MPI_VDA_CD_Init]failed number error\n");
        return FY_FAILURE;
    }

    id = pCfg->ID;
    if (!g_CoverCfgInfo[id].bInit) {
        memset(&g_CoverCfgInfo[id], 0, sizeof(FY_VDA_CD_CFG));
        if (pthread_mutex_init(&g_CoverCfgInfo[id].cdMutex, 0)) {
            return FY_FAILURE;
        }
    } else {
        return FY_SUCCESS;
    }
    show_MD_VERSION(id);
    memcpy(&g_CoverCfgInfo[id].initParam, pCfg, sizeof(FY_VDA_INIT_cfg));
    if (g_CoverCfgInfo[id].initParam.threadMode == DRV_THREAD_MODE) {
        VDA_Create_thread(type, 75);
    } else {
        if (g_VdaThrInfo[type].t_id != 0) {
            g_VdaThrInfo[type].bStart = FY_FALSE;
            pthread_join(g_VdaThrInfo[type].t_id, 0);
            g_VdaThrInfo[type].t_id = 0;
        }
    }
    pthread_mutex_lock(&g_CoverCfgInfo[id].cdMutex);
    g_CoverCfgInfo[id].bInit = FY_TRUE;
    sprintf(g_CoverCfgInfo[id].Yc.name, "VDA-CD-Yc-%d", id);
    pthread_mutex_unlock(&g_CoverCfgInfo[id].cdMutex);

    return FY_SUCCESS;
}

FY_S32 MPI_VDA_CD_DeInit(FY_U32 id) {
    CHECK_ID_IS_VALID(id);
    pthread_mutex_lock(&g_CoverCfgInfo[id].cdMutex);
    g_CoverCfgInfo[id].bInit = FY_FALSE;
    g_CoverCfgInfo[id].cdCfg.enable = FY_FALSE;
    VDA_Free_Memory(g_CoverCfgInfo[id].Yc.pVirAddr);
    pthread_mutex_unlock(&g_CoverCfgInfo[id].cdMutex);
    pthread_mutex_destroy(&g_CoverCfgInfo[id].cdMutex);
    memset(&g_CoverCfgInfo[id], 0, sizeof(FY_VDA_CD_CFG));
    return FY_SUCCESS;
}

FY_S32 MPI_VDA_CD_Update_BG(FY_U32 id, FY_S32 vpssGrp, FY_S32 vpassChn) {
    FY_S32 ret = FY_SUCCESS;

    return ret;
}

FY_S32 MPI_VDA_CD_SetConfig(FY_VDA_CD_INFO_t* pCDCfg) {
    FY_U32 id = 0;

    VDA_CHECK_PTR_NULL(pCDCfg);
    id = pCDCfg->ID;
    CHECK_ID_IS_VALID(id);
    VDA_CD_IS_INIT(id);

    pthread_mutex_lock(&g_CoverCfgInfo[id].cdMutex);
    g_CoverCfgInfo[id].cdCfg.enable = pCDCfg->enable;
    g_CoverCfgInfo[id].cdCfg.framedelay = pCDCfg->framedelay;
    g_CoverCfgInfo[id].cdCfg.level = pCDCfg->level;
    g_CoverCfgInfo[id].cdCfg.changevalue = pCDCfg->changevalue;
    pthread_mutex_unlock(&g_CoverCfgInfo[id].cdMutex);

    return FY_SUCCESS;
}

FY_S32 MPI_VDA_CD_GetConfig(FY_VDA_CD_INFO_t* pCDCfg) {
    FY_U32 id = 0;

    VDA_CHECK_PTR_NULL(pCDCfg);
    id = pCDCfg->ID;
    CHECK_ID_IS_VALID(id);
    VDA_CD_IS_INIT(id);

    pthread_mutex_lock(&g_CoverCfgInfo[id].cdMutex);
    pCDCfg->enable = g_CoverCfgInfo[id].cdCfg.enable;
    pCDCfg->framedelay = g_CoverCfgInfo[id].cdCfg.framedelay;
    pCDCfg->level = g_CoverCfgInfo[id].cdCfg.level;
    pCDCfg->changevalue = g_CoverCfgInfo[id].cdCfg.changevalue;
    pthread_mutex_unlock(&g_CoverCfgInfo[id].cdMutex);

    return FY_SUCCESS;
}

FY_S32 MPI_VDA_CD_GetResult(FY_U32 id, FY_U32* pCDStatus) {
    CHECK_ID_IS_VALID(id);
    VDA_CD_IS_INIT(id);
    VDA_CHECK_PTR_NULL(pCDStatus);

    pthread_mutex_lock(&g_CoverCfgInfo[id].cdMutex);
    if (g_CoverCfgInfo[id].cdCfg.enable) {
        *pCDStatus = g_CoverCfgInfo[id].cdResult;
    } else {
        *pCDStatus = 0;
    }
    pthread_mutex_unlock(&g_CoverCfgInfo[id].cdMutex);
    return FY_SUCCESS;
}

FY_S32 MPI_VDA_CD_GetConstancy(FY_U32 id, FY_Rect_t* pRect, FY_VOID* pConstan) {
    FY_BOOL bLog = FY_FALSE;
    FY_S32 ret = FY_SUCCESS;

    CHECK_ID_IS_VALID(id);
    VDA_CD_IS_INIT(id);

    pthread_mutex_lock(&g_CoverCfgInfo[id].cdMutex);
    bLog = g_CoverCfgInfo[id].initParam.bOpenLog;
    VDA_LOG(bLog, "[%s][CD%d]width=%d,height=%d\n", __func__, id, g_CoverCfgInfo[id].u32Width,
            g_CoverCfgInfo[id].u32Height);
    if (pConstan) {
        memcpy(pConstan, &g_CoverCfgInfo[id].count[0], MAX_CNT * sizeof(FY_U32));
    }
    if (pRect) {
        pRect->fTopLeftX = 0;
        pRect->fTopLeftY = 0;
        pRect->fWidth = g_CoverCfgInfo[id].u32Width;
        pRect->fHeigh = g_CoverCfgInfo[id].u32Height;
    }
    pthread_mutex_unlock(&g_CoverCfgInfo[id].cdMutex);
    return ret;
}

FY_S32 FY_IOE_Open(void) {
    FY_S32 result = FY_SUCCESS;

    if (!g_VdaHwInfo.bOpen) {
        g_VdaHwInfo.sFd = open("/dev/mcioe", O_RDWR, 0);  // close
        if (g_VdaHwInfo.sFd < 0) {
            printf("\033[31m[%s]open ioe fail\033[0m\n", __func__);
            result = FY_FAILURE;
        } else {
            g_VdaHwInfo.bOpen = FY_TRUE;
        }
    }
    return result;
}

FY_S32 FY_IOE_OPERATE(IOE_PARAM_T* ioe_param) {
    FY_S32 result = 0;

    VDA_CHECK_PTR_NULL(ioe_param);
    VDA_CHECK_PTR_NULL(g_VdaHwInfo.bOpen);

    result = ioctl(g_VdaHwInfo.sFd, IOC_IOE_OPERATE, ioe_param);
    if (result) {
        printf("[%s]opreate error!\n", __func__);
    }
    return result;
}

FY_S32 FY_IOE_Close(void) {
    FY_S32 result = FY_SUCCESS;

    if (g_VdaHwInfo.bOpen) {
        close(g_VdaHwInfo.sFd);
    }
    memset(&g_VdaHwInfo, 0, sizeof(g_VdaHwInfo));
    return result;
}
