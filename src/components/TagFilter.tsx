'use client';

interface TagFilterProps {
  availableTags: { tag: string; count: number }[];
  selectedTags: string[];
  onTagToggle: (tag: string) => void;
}

export default function TagFilter({ availableTags, selectedTags, onTagToggle }: TagFilterProps) {
  if (availableTags.length === 0) {
    return null;
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h3 className="text-sm font-medium text-gray-700 mb-3">Filter by tags:</h3>
      <div className="flex flex-wrap gap-2">
        {availableTags.map(({ tag, count }) => {
          const isSelected = selectedTags.includes(tag);
          return (
            <button
              key={tag}
              onClick={() => onTagToggle(tag)}
              className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                isSelected
                  ? 'bg-blue-600 text-white hover:bg-blue-700'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              {tag}
              <span className={`ml-1 text-xs ${
                isSelected ? 'text-blue-200' : 'text-gray-500'
              }`}>
                ({count})
              </span>
            </button>
          );
        })}
      </div>
      
      {selectedTags.length > 0 && (
        <div className="mt-3">
          <button
            onClick={() => selectedTags.forEach(tag => onTagToggle(tag))}
            className="text-sm text-gray-500 hover:text-gray-700 underline"
          >
            Clear all tags
          </button>
        </div>
      )}
    </div>
  );
}