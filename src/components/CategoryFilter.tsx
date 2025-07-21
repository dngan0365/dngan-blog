'use client';

interface CategoryFilterProps {
  availableCategories: { category: string; count: number }[];
  selectedCategory: string | null;
  onCategorySelect: (category: string | null) => void;
}

export default function CategoryFilter({ 
  availableCategories, 
  selectedCategory, 
  onCategorySelect 
}: CategoryFilterProps) {
  if (availableCategories.length === 0) {
    return null;
  }

  return (
    <div className="max-w-4xl mx-auto mb-6">
      <h3 className="text-sm font-medium text-gray-700 mb-3">Browse by category:</h3>
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => onCategorySelect(null)}
          className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium transition-colors ${
            selectedCategory === null
              ? 'bg-blue-600 text-white hover:bg-blue-700'
              : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
          }`}
        >
          All Categories
          <span className={`ml-2 text-xs ${
            selectedCategory === null ? 'text-blue-200' : 'text-gray-500'
          }`}>
            ({availableCategories.reduce((sum, cat) => sum + cat.count, 0)})
          </span>
        </button>
        
        {availableCategories.map(({ category, count }) => (
          <button
            key={category}
            onClick={() => onCategorySelect(category)}
            className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-medium transition-colors ${
              selectedCategory === category
                ? 'bg-blue-600 text-white hover:bg-blue-700'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            {category}
            <span className={`ml-2 text-xs ${
              selectedCategory === category ? 'text-blue-200' : 'text-gray-500'
            }`}>
              ({count})
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}