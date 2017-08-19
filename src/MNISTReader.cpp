#include "MNISTReader.h"

#include <fstream>
using namespace std;

//TODO clean everything , fix warnings (at least it works)

////////////////////////////////////////////////////////////////////////////////////
bool MNISTReader::read_from_folder(const string& sFolder,MatrixFloat& mRefImages,MatrixFloat& mRefLabels,MatrixFloat& mTestImages,MatrixFloat& mTestLabels)
{
    string sRefImages=sFolder+"\\train-images.idx3-ubyte";
    string sRefLabels=sFolder+"\\train-labels.idx1-ubyte";
    string sTestImages=sFolder+"\\t10k-images.idx3-ubyte";
    string sTestLabels=sFolder+"\\t10k-labels.idx1-ubyte";

    if(!read_Matrix(sRefImages, mRefImages))
        return false;

    if(!read_Matrix(sRefLabels, mRefLabels))
        return false;

    if(!read_Matrix(sTestImages, mTestImages))
        return false;

    if(!read_Matrix(sTestLabels, mTestLabels))
        return false;

    return true;
}
////////////////////////////////////////////////////////////////////////////////////
bool MNISTReader::read_Matrix(string sName,MatrixFloat& m)
{
    // file format and data at : http://yann.lecun.com/exdb/mnist/

    ifstream ifs(sName, ios::binary|ios::in );

    if(!ifs)
        return false;

    short usMagic;
    ifs.read((char*)(&usMagic),2);
    if(usMagic!=0)
        return false;

    char ucType;
    ifs.read((char*)(&ucType),1);
    if(ucType!=0x08)
        return false; //only byte format for now

    unsigned char ucNbDim;
    ifs.read((char*)(&ucNbDim),1);
    if(ucNbDim==0x01)
    {
        // one vector data
        unsigned int uiSize;
        ifs.read((char*)(&uiSize),4); swap_int(uiSize);

        m.resize(uiSize,1);

        char* pVector=new char[uiSize];
        ifs.read((char*)(pVector),uiSize);
        float* pData=m.data();
        for(unsigned int i=0;i<uiSize;i++)
        {
            *pData++=pVector[i];
        }

        delete[] pVector;
        return true;
    }

    if(ucNbDim==0x03)
    {
        // image list data , will be flattened, one row by image
        unsigned int uiNbImages,uiNbRows,uiNbColumns;
        ifs.read((char*)(&uiNbImages),4); swap_int(uiNbImages);
        ifs.read((char*)(&uiNbRows),4); swap_int(uiNbRows);
        ifs.read((char*)(&uiNbColumns),4); swap_int(uiNbColumns);

        unsigned int uiSize=uiNbRows*uiNbColumns;

        m.resize(uiNbImages,uiSize);

        char* pImage=new char[uiSize];

        for(unsigned int iImage=0;iImage<uiNbImages;iImage++)
        {
            ifs.read(pImage,uiSize);
            float * pData=m.row(iImage).data();
            for(unsigned int i=0;i<uiSize;i++)
            {
                *pData++=pImage[i];
            }
        }

        delete[] pImage;
        return true;
    }

    return false;
}
////////////////////////////////////////////////////////////////////////////////////
void MNISTReader::swap_int(unsigned int & i)
{
    unsigned char c1=(unsigned char)(i & 0xFF);
    unsigned char c2=(unsigned char)((i >> 8) & 0xFF);
    unsigned char c3=(unsigned char)((i >>16 )& 0xFF);
    unsigned char c4=(unsigned char)((i>> 24) & 0xFF);

    i=(c1<<24)+(c2<<16)+(c3<<8)+c4;
}
////////////////////////////////////////////////////////////////////////////////////
