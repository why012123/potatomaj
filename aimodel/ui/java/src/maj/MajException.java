package maj;

/** 和后端说话时出的任何问题（连不上、超时、HTTP 错误、JSON 坏了）。 */
public class MajException extends Exception {

    private static final long serialVersionUID = 1L;

    public MajException(String message) {
        super(message);
    }

    public MajException(String message, Throwable cause) {
        super(message, cause);
    }
}
