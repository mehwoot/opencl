#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <tr1/functional>

#ifdef __APPLE__
        #include "OpenCL/opencl.h"
#else
        #include "CL/cl.h"
#endif

std::string GetPlatformName (cl_platform_id id)
{
        size_t size = 0;
        clGetPlatformInfo (id, CL_PLATFORM_NAME, 0, NULL, &size);

        std::string result;
        result.resize (size);
        clGetPlatformInfo (id, CL_PLATFORM_NAME, size,
                const_cast<char*> (result.data ()), NULL);

        return result;
}

std::string GetDeviceName (cl_device_id id)
{
        size_t size = 0;
        clGetDeviceInfo (id, CL_DEVICE_NAME, 0, NULL, &size);

        std::string result;
        result.resize (size);
        clGetDeviceInfo (id, CL_DEVICE_NAME, size,
                const_cast<char*> (result.data ()), NULL);

        return result;
}

void CheckError (cl_int error)
{
        if (error != CL_SUCCESS) {
                std::cerr << "OpenCL call failed with error " << error << std::endl;
                std::exit (1);
        }
}

std::string LoadKernel (const char* name)
{
        std::ifstream in (name);
        std::string result (
                (std::istreambuf_iterator<char> (in)),
                std::istreambuf_iterator<char> ());
        return result;
}

cl_program CreateProgram (const std::string& source,
        cl_context context)
{
        size_t lengths [1] = { source.size () };
        const char* sources [1] = { source.data () };

        cl_int error = 0;
        cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
        CheckError (error);

        return program;
}

int main (int argv, char** argv)
{
        // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetPlatformIDs.html
        cl_uint platformIdCount = 0;
        clGetPlatformIDs (0, NULL, &platformIdCount);

        if (platformIdCount == 0) {
                std::cerr << "No OpenCL platform found" << std::endl;
                return 1;
        } else {
                std::cout << "Found " << platformIdCount << " platform(s)" << std::endl;
        }

        std::vector<cl_platform_id> platformIds (platformIdCount);
        clGetPlatformIDs (platformIdCount, platformIds.data (), NULL);

        for (cl_uint i = 0; i < platformIdCount; ++i) {
                std::cout << "\t (" << (i+1) << ") : " << GetPlatformName (platformIds [i]) << std::endl;
        }

        // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clGetDeviceIDs.html
        cl_uint deviceIdCount = 0;
        clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_GPU, 0, NULL,
                &deviceIdCount);

        if (deviceIdCount == 0) {
                std::cerr << "No OpenCL devices found" << std::endl;
                return 1;
        } else {
                std::cout << "Found " << deviceIdCount << " device(s)" << std::endl;
        }

        std::vector<cl_device_id> deviceIds (deviceIdCount);
        clGetDeviceIDs (platformIds [0], CL_DEVICE_TYPE_GPU, deviceIdCount,
                deviceIds.data (), NULL);

        for (cl_uint i = 0; i < deviceIdCount; ++i) {
                std::cout << "\t (" << (i+1) << ") : " << GetDeviceName (deviceIds [i]) << std::endl;
        }

        // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateContext.html
        const cl_context_properties contextProperties [] =
        {
                CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties> (platformIds [0]),
                0, 0
        };

        cl_int error = CL_SUCCESS;
        cl_context context = clCreateContext (contextProperties, deviceIdCount,
                deviceIds.data (), NULL, NULL, &error);
        CheckError (error);

        std::cout << "Context created" << std::endl;

        cl_program program = CreateProgram (LoadKernel ("saxpy.cl"),
                context);

        CheckError (clBuildProgram (program, deviceIdCount, deviceIds.data (), NULL, NULL, NULL));

        cl_kernel kernel = clCreateKernel (program, "SAXPY", &error);
        CheckError (error);

        // Prepare some test data
        static const size_t testDataSize = 2;
        std::vector<float> x (testDataSize), y (testDataSize), c (testDataSize), z(testDataSize);
        x[0] = 3.1;
        x[1] = 1.453f;
        y[0] = 3.3453453f;
        y[1] = 7.29009234f;

        for (int i=0; i<2; i++) {
          z [i] = x[i] + y[i];
          for (int j=0; j<100; j++) {
            /*z[i] = sqrt(z[i]);*/
            z[i] = z[i] + y[i];
            z[i] = z[i] / x[i];
          }
        }

        cl_mem aBuffer = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                sizeof (float) * (testDataSize),
                x.data (), &error);
        CheckError (error);

        cl_mem bBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                sizeof (float) * (testDataSize),
                y.data (), &error);
        CheckError (error);

        cl_mem cBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                sizeof (float) * (testDataSize),
                c.data (), &error);
        CheckError (error);

        // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateCommandQueue.html
        cl_command_queue queue = clCreateCommandQueue (context, deviceIds [0],
                0, &error);
        CheckError (error);

        clSetKernelArg (kernel, 0, sizeof (cl_mem), &aBuffer);
        clSetKernelArg (kernel, 1, sizeof (cl_mem), &bBuffer);
        clSetKernelArg (kernel, 2, sizeof (cl_mem), &cBuffer);
        /*static const float two = 2.0f;
        clSetKernelArg (kernel, 2, sizeof (float), &two);*/

        // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueNDRangeKernel.html
        const size_t globalWorkSize [] = { testDataSize, 0, 0 };
        CheckError (clEnqueueNDRangeKernel (queue, kernel, 1,
                NULL,
                globalWorkSize,
                NULL,
                0, NULL, NULL));

        // Get the results back to the host
        // http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clEnqueueReadBuffer.html
        CheckError (clEnqueueReadBuffer (queue, cBuffer, CL_TRUE, 0,
                sizeof (float) * testDataSize,
                c.data (),
                0, NULL, NULL));

        std::tr1::hash<float> hasher;

        for (int i=0; i<testDataSize; i++) {
          printf("Value %d: %f, CPU: %f\n", i, c[i], z[i]);
          printf("Value Hash %d: %d, CPU: %d\n", i, (int)hasher(c[i]), (int)hasher(z[i]));
        }

        clReleaseCommandQueue (queue);

        clReleaseMemObject (bBuffer);
        clReleaseMemObject (aBuffer);

        clReleaseKernel (kernel);
        clReleaseProgram (program);

        clReleaseContext (context);
}