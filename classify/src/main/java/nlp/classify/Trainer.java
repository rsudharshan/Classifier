package nlp.classify;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.hadoop.fs.Path;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.util.Version;
import org.apache.mahout.classifier.BayesFileFormatter;
import org.apache.mahout.classifier.bayes.TrainClassifier;
import org.apache.mahout.classifier.bayes.mapreduce.cbayes.CBayesDriver;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.classifier.bayes.BayesParameters;
/**
 *
 * @author Sudharshan
 */
public class Trainer {
    /**
     * A program to automate pre-processing , training of models in Classifier
     *
     * @param path input source documents
     * @param outputPath processed document store
     */
    public static BayesParameters params = new BayesParameters();
    public static Properties props = new Properties();

    public static void prepareClassDirectory(String path, String outputPath) {
        File[] f = new File(path).listFiles();
        for (File s : f) {
            if (s.isDirectory()) {
                String label = s.getName();

                System.out.println(label);
                File idir = new File(s.getAbsolutePath());
                File odir = new File(outputPath + label + ".txt");
                File oPath = new File(outputPath);

                Analyzer al = new StandardAnalyzer(Version.LUCENE_CURRENT);
                Charset cr = Charset.forName("UTF-8");
                try {
                    BayesFileFormatter.collapse(label, al, idir, cr, odir);
                    /*
                     */
                } catch (IOException ex) {
                    Logger.getLogger(TrainClassifier.class.getName()).log(Level.SEVERE, null, ex);
                }

                System.out.println("Finished Writing" + s);

            }

        }
    }
 /**
     *
     * @param fnDocument
     * @return Processed Text (stop words removed)
     * @throws IOException
     */
    public static String preprocessDocument(File fnDocument)
            throws IOException {
        StandardAnalyzer analyzer = new StandardAnalyzer(Version.LUCENE_30);
        TokenStream tokenstream = analyzer.tokenStream(null, new InputStreamReader(new FileInputStream(fnDocument), "UTF-8"));

        List<String> tokenlist = new ArrayList<String>();

        while (tokenstream.incrementToken()) {
            tokenlist.add(tokenstream.getAttribute(TermAttribute.class).term());
        }
        String[] preparedDoc = tokenlist.toArray(new String[tokenlist.size()]);
        String t = getSingleLineText(preparedDoc);
        return t;
    }
    /**
     *
     * @param toks token stream
     * @return single line of text
     */
    public static String getSingleLineText(String[] toks) {
        String text = null;
        StringBuilder sb = new StringBuilder();
        for (String s : toks) {
            sb.append(s + " ");
        }

        text = sb.toString();
        return text;
    }   
    
    /**
     *
     * @param ipdir
     * @param opmodeldir
     */
    
    public static void trainCBayesAlgorithm(String ipdir, String opmodeldir) {
        Path ip = new Path(ipdir);
        Path op = new Path(opmodeldir);       
        try {
            CBayesDriver driver = new CBayesDriver();
            driver.runJob(ip, op, params);
        } catch (IOException ex) {
            Logger.getLogger(Trainer.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
	public static void initializeDefaults() throws Exception {
		String fileName = "settings.properties";
		InputStream inputStream = new FileInputStream(fileName);
		props.load(inputStream);
	}
    /**
     *
     * @param a
     */
    public static void main(String a[]) {
    
    	try {
			initializeDefaults();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	//try {
        String path = props.getProperty("TrainSet");//"/home/developer/dataset_rev/freshrevs/train/";
        String opPath = props.getProperty("ProcessedSet");//"/home/developer/dataset_rev/freshrevs/processedTrain/";
        String modelPath = props.getProperty("ModelPath");//"/home/developer/dataset_rev/freshrevs/model/";
        params.setGramSize(Integer.parseInt(props.getProperty("GramSize")));
        params.set("classifierType", props.getProperty("cbayes"));
        params.set("dataSource", props.getProperty("DataSource"));
        params.set("defaultCat", props.getProperty("DefaultCategory"));
        params.set("encoding", props.getProperty("Encoding"));
        params.set("alpha_i", props.getProperty("Alpha"));
       // prepareTweetClassDirectory(path, opPath);
        prepareClassDirectory(path, opPath);      
        trainCBayesAlgorithm(opPath, modelPath);
       /*String txt = CBayesPredictor.normalizeSourceText("bmth show saturday :] leather jackets skinny blue jeans heels, city loves us tonight :) arun shourie rocks bjp. again. !!! i am loving it! !! doesn't make me congress man. !!! :) ");
            System.out.println(txt);
        } catch (Exception ex) {
            Logger.getLogger(Trainer.class.getName()).log(Level.SEVERE, null, ex);
        }*/
    }
}
