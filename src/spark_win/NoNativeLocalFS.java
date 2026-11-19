package bft;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RawLocalFileSystem;
import org.apache.hadoop.fs.permission.FsPermission;

/**
 * Windows only. Hadoop's local file system calls hadoop.dll (NativeIO.Windows.access0) to list a
 * folder, and Apache does not ship that DLL. This class lists with java.io.File instead, so Spark
 * reads Parquet on Windows without third-party binaries. Reading data is unchanged: Hadoop's own
 * RawLocalFileSystem still opens the files. Not used on Linux, macOS or Colab.
 */
public class NoNativeLocalFS extends RawLocalFileSystem {
    @Override
    public FileStatus[] listStatus(Path f) throws IOException {
        File local = pathToFile(f);
        if (!local.exists()) throw new FileNotFoundException("File " + f + " does not exist");
        if (local.isFile()) return new FileStatus[] {getFileStatus(f)};
        String[] names = local.list();
        if (names == null) throw new IOException("Cannot list " + f);
        FileStatus[] out = new FileStatus[names.length];
        for (int i = 0; i < names.length; i++) out[i] = getFileStatus(new Path(f, names[i]));
        return out;
    }

    @Override
    public void setPermission(Path p, FsPermission permission) {
        // Windows does not use POSIX permissions; Hadoop would call winutils.exe here.
    }
}
