using System;
using Emgu.CV;
using System.IO;

namespace FacialDetection
{
    class Program
    {
        static void Main(string[] args)
        {
            var faceClassifier = new CascadeClassifier(Path.Join("resources", 
                "haarcascade_frontalface_default.xml"));
            var img = CvInvoke.Imread(Path.Join("resources", "imageWithFace.jpg"));

            var faces = faceClassifier.DetectMultiScale(img,
                minSize: new System.Drawing.Size(300,300));
            foreach(var face in faces)
            {
                CvInvoke.Rectangle(img, face, 
                    new Emgu.CV.Structure.MCvScalar(255, 0, 0), 10);
            }

            CvInvoke.Imshow("Faces", img);
            CvInvoke.WaitKey(0);
        }
    }
}
