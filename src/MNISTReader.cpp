#include "MNISTReader.h"

#include <fstream>
using namespace std;

////////////////////////////////////////////////////////////////////////////////////
bool MNISTReader::read_from_folder(const string& sFolder,Matrix& mRefImages,Matrix& mRefLabels,Matrix& mTestImages,Matrix& mTestLabels)
{
    string sRefImages=sFolder+"\\train-images.idx3-ubyte";
    string sRefLabels=sFolder+"\\train-labels.idx1-ubyte";
    string sTestImages=sFolder+"\\t10k-images.idx3-ubyte";
    string sTestLabels=sFolder+"\\t10k-labels.idx1-ubyte";

    if(!read_matrix(sRefImages, mRefImages))
        return false;

    if(!read_matrix(sRefLabels, mRefLabels))
        return false;

    if(!read_matrix(sTestImages, mTestImages))
        return false;

    if(!read_matrix(sTestLabels, mTestLabels))
        return false;

    return true;
}
////////////////////////////////////////////////////////////////////////////////////
bool MNISTReader::read_matrix(string sName,Matrix& m)
{
    // file format and data at : http://yann.lecun.com/exdb/mnist/

    fstream ifs(sName,ios::binary);
    if(!ifs)
        return false;

    unsigned short usMagic;
    ifs >> usMagic;
    if(usMagic!=0)
        return false;

    unsigned char ucType;
    ifs >> ucType;
    if(ucType!=0x08)
        return false; //only byte format for now

    unsigned char ucNbDim;
    ifs >> ucNbDim;
    if(ucNbDim==0x01)
    {
        // one vector data
        unsigned int uiSize;
        ifs >> uiSize;

        m.resize(1,uiSize);

        for(unsigned int i=0;i<uiSize;i++)
        {
            unsigned char ucData; //todo: read batch
            ifs >> ucData;
            m(i)=ucData;
        }

        return true;
    }

    if(ucNbDim==0x03)
    {
        // image list data , will be flattened, one row by image
        unsigned int uiNbImages,uiNbRows,uiNbColumns;
        ifs >> uiNbImages >> uiNbRows >> uiNbColumns;
        unsigned int uiSize=uiNbRows*uiNbColumns;

        m.resize(uiNbImages,uiSize);
        for(unsigned int iImage=0;iImage<uiNbImages;iImage++)
        {
            double * pData=m.row(iImage).data();
            for(unsigned int i=0;i<uiSize;i++)
            {
                unsigned char ucData; //todo: read batch
                ifs >> ucData;
                *pData++=ucData;
            }
        }

        return true;
    }

    return false;
}
////////////////////////////////////////////////////////////////////////////////////
