export default function Header() {
  return (
    <header className="bg-background-light dark:bg-background-dark p-4 flex items-center shadow-sm">
      <button className="text-gray-800 dark:text-gray-200">
        <span className="material-symbols-outlined">arrow_back_ios_new</span>
      </button>
      <h1 className="flex-1 text-center text-lg font-bold text-gray-900 dark:text-white pr-6">
        Upload Configuration
      </h1>
    </header>
  );
}
