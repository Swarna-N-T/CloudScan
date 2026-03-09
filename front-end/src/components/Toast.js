export function ToastSuccess() {
  return (
    <div className="fixed bottom-24 left-1/2 -translate-x-1/2 flex items-center w-full max-w-xs p-4 text-gray-500 bg-white rounded-lg shadow dark:text-gray-400 dark:bg-gray-800">
      <div className="text-green-500">
        <span className="material-symbols-outlined">check_circle</span>
      </div>
      <div className="pl-4 text-sm font-normal">Upload successful.</div>
    </div>
  );
}

export function ToastError() {
  return (
    <div className="fixed bottom-24 left-1/2 -translate-x-1/2 flex items-center w-full max-w-xs p-4 text-gray-500 bg-white rounded-lg shadow dark:text-gray-400 dark:bg-gray-800">
      <div className="text-red-500">
        <span className="material-symbols-outlined">cancel</span>
      </div>
      <div className="pl-4 text-sm font-normal">Upload failed. Please try again.</div>
    </div>
  );
}

/* 👇 ADD THIS WRAPPER COMPONENT */
export function Toast({ type }) {
  if (type === "success") return <ToastSuccess />;
  if (type === "error") return <ToastError />;
  return null;
}
